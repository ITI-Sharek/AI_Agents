from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from sharek_agents.agents.document_understanding.schemas import CloudinaryResourceRef


# ── Supported resource types & formats ────────────────────────────────────────

SUPPORTED_RESOURCE_TYPES: frozenset[str] = frozenset({"raw"})

SUPPORTED_DOCUMENT_FORMATS: frozenset[str] = frozenset({
    "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx",
    "txt", "rtf", "odt",
    "csv", "json", "xml", "yaml", "yml",
    "md", "html", "htm",
})

FORMAT_TO_MIME: dict[str, str] = {
    "pdf": "application/pdf",
    "doc": "application/msword",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xls": "application/vnd.ms-excel",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "ppt": "application/vnd.ms-powerpoint",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "txt": "text/plain",
    "rtf": "application/rtf",
    "odt": "application/vnd.oasis.opendocument.text",
    "csv": "text/csv",
    "json": "application/json",
    "xml": "application/xml",
    "yaml": "application/x-yaml",
    "yml": "application/x-yaml",
    "md": "text/markdown",
    "html": "text/html",
    "htm": "text/html",
}


# ── Error hierarchy ───────────────────────────────────────────────────────────


class CloudinaryError(Exception):
    """Base error for all Cloudinary operations."""


class CloudinaryConfigError(CloudinaryError):
    """Cloudinary is not configured or configuration is incomplete."""


class CloudinaryAuthError(CloudinaryError):
    """Cloudinary authentication rejected the provided credentials."""


class CloudinaryNotFoundError(CloudinaryError):
    """The requested resource does not exist on Cloudinary."""


class CloudinaryUnsupportedTypeError(CloudinaryError):
    """The resource type is not supported for document retrieval."""


class CloudinaryUnsupportedFormatError(CloudinaryError):
    """The file format is not supported for document processing."""


class CloudinaryDownloadError(CloudinaryError):
    """The resource content could not be downloaded."""


class CloudinaryNetworkError(CloudinaryError):
    """A network-level error occurred while communicating with Cloudinary."""


class CloudinaryFileSizeLimitError(CloudinaryError):
    """The resource exceeds the maximum allowed file size."""


# ── Retrieved document model ──────────────────────────────────────────────────


@dataclass(frozen=True)
class RetrievedDocument:
    """Normalized representation of a successfully retrieved Cloudinary document."""
    reference: CloudinaryResourceRef
    filename: str
    content_type: str
    resource_type: str
    file_format: str | None
    file_size: int
    content: bytes
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class CloudinaryConfig:
    cloud_name: str
    api_key: str
    api_secret: str
    max_file_size_bytes: int = 50 * 1024 * 1024
    default_resource_type: str = "raw"
    timeout_seconds: float = 120.0

    @property
    def is_configured(self) -> bool:
        return bool(self.cloud_name and self.api_key and self.api_secret)


# ── Protocol (public interface, easy to mock) ─────────────────────────────────


class CloudinaryClient(Protocol):
    """Interface for retrieving documents from Cloudinary.

    Implementations must not hard-code credentials and must not
    expose credentials in exceptions or log output.
    """

    async def retrieve(self, ref: CloudinaryResourceRef) -> RetrievedDocument:
        """Download a document identified by *ref*.

        Raises:
            CloudinaryConfigError: Credentials are missing.
            CloudinaryAuthError: Authentication failed.
            CloudinaryNotFoundError: Resource does not exist.
            CloudinaryUnsupportedTypeError: Resource type not supported.
            CloudinaryUnsupportedFormatError: File format not supported.
            CloudinaryFileSizeLimitError: File exceeds size limit.
            CloudinaryDownloadError: Download failed.
            CloudinaryNetworkError: Network error.
        """
        ...


# ── Concrete implementation ───────────────────────────────────────────────────


class CloudinaryClientImpl:
    """Retrieves documents from Cloudinary via the Admin API and CDN URLs.

    Credentials are read from the supplied *config* — never hard-coded.

    An optional ``http_client`` can be injected for testing.  When provided
    it is used directly (without context-manager lifecycle) so that tests
    can pass a mock without fighting AsyncMock's auto-creation behaviour.
    """

    _ADMIN_API = "https://api.cloudinary.com/v1_1/{cloud_name}/resources/{resource_type}/{delivery_type}"

    def __init__(self, config: CloudinaryConfig, http_client: httpx.AsyncClient | None = None) -> None:
        self._config = config
        self._http_client = http_client

    async def retrieve(self, ref: CloudinaryResourceRef) -> RetrievedDocument:
        self._assert_configured()

        if ref.public_id:
            return await self._retrieve_by_public_id(ref)
        if ref.url:
            return await self._retrieve_by_url(ref)
        raise CloudinaryError("Resource reference has neither public_id nor url")

    # ── Internals ─────────────────────────────────────────────────────────────

    def _assert_configured(self) -> None:
        if not self._config.is_configured:
            raise CloudinaryConfigError(
                "Cloudinary is not configured. "
                "Set CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, "
                "and CLOUDINARY_API_SECRET."
            )

    @staticmethod
    def _safe_message(msg: str) -> str:
        """Return a message that does not leak credentials or secrets."""
        return msg

    # -- Retrieval by public_id via Admin API --

    async def _retrieve_by_public_id(self, ref: CloudinaryResourceRef) -> RetrievedDocument:
        resource_type = ref.resource_type or self._config.default_resource_type
        delivery_type = ref.delivery_type or "upload"

        resource = await self._admin_api_resource(ref.public_id, resource_type, delivery_type)

        actual_resource_type: str = resource.get("resource_type", resource_type)
        self._validate_resource_type(actual_resource_type)

        file_format: str | None = (
            ref.format
            or resource.get("format")
            or self._extract_format(ref.public_id)
        )
        self._validate_format(file_format)

        resource_size: int = resource.get("bytes", 0)
        self._validate_size(resource_size)

        secure_url: str | None = resource.get("secure_url")
        if not secure_url:
            raise CloudinaryDownloadError(self._safe_message(
                "Cloudinary response did not include a download URL"
            ))

        content = await self._download_content(secure_url)

        filename = self._build_filename(ref.public_id, file_format)
        content_type = FORMAT_TO_MIME.get(file_format or "", "application/octet-stream")

        return RetrievedDocument(
            reference=ref,
            filename=filename,
            content_type=content_type,
            resource_type=actual_resource_type,
            file_format=file_format,
            file_size=len(content),
            content=content,
            metadata=self._build_safe_metadata(resource),
        )

    async def _admin_api_resource(
        self,
        public_id: str,
        resource_type: str,
        delivery_type: str,
    ) -> dict[str, Any]:
        url = (
            f"https://api.cloudinary.com/v1_1/{self._config.cloud_name}"
            f"/resources/{resource_type}/{delivery_type}/{public_id}"
        )
        auth = httpx.BasicAuth(self._config.api_key, self._config.api_secret)

        try:
            response = await self._http_get(url, auth=auth)
        except httpx.TimeoutException as exc:
            raise CloudinaryNetworkError(
                self._safe_message("Cloudinary Admin API request timed out")
            ) from exc
        except httpx.RequestError as exc:
            raise CloudinaryNetworkError(
                self._safe_message("Network error contacting Cloudinary Admin API")
            ) from exc

        if response.status_code == 401:
            raise CloudinaryAuthError(
                self._safe_message("Cloudinary authentication failed — check API key and secret")
            )
        if response.status_code == 403:
            raise CloudinaryAuthError(
                self._safe_message("Cloudinary credentials are not authorized for this operation")
            )
        if response.status_code == 404:
            raise CloudinaryNotFoundError(
                self._safe_message(
                    f"Cloudinary resource not found: {public_id}"
                )
            )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise CloudinaryDownloadError(
                self._safe_message(f"Cloudinary Admin API returned status {exc.response.status_code}")
            ) from exc

        return response.json()

    # -- Retrieval by URL --

    async def _retrieve_by_url(self, ref: CloudinaryResourceRef) -> RetrievedDocument:
        content = await self._download_content(ref.url)
        self._validate_size(len(content))

        file_format: str | None = ref.format or self._guess_format_from_url(ref.url)
        self._validate_format(file_format)

        filename = self._guess_filename_from_url(ref.url) or "document"
        content_type = (
            FORMAT_TO_MIME.get(file_format or "")
            or ref.mime_type
            or "application/octet-stream"
        )

        return RetrievedDocument(
            reference=ref,
            filename=filename,
            content_type=content_type,
            resource_type="raw",
            file_format=file_format,
            file_size=len(content),
            content=content,
            metadata={},
        )

    # -- Download --

    async def _http_get(self, url: str, **kwargs: Any) -> httpx.Response:
        """Perform an HTTP GET, optionally using an injected client."""
        if self._http_client is not None:
            return await self._http_client.get(url, **kwargs)
        async with httpx.AsyncClient(timeout=self._config.timeout_seconds) as client:
            return await client.get(url, **kwargs)

    async def _download_content(self, url: str) -> bytes:
        try:
            response = await self._http_get(url, follow_redirects=True)
        except httpx.TimeoutException as exc:
            raise CloudinaryNetworkError(
                self._safe_message("Download timed out")
            ) from exc
        except httpx.RequestError as exc:
            raise CloudinaryNetworkError(
                self._safe_message("Network error during download")
            ) from exc

        if response.status_code == 404:
            raise CloudinaryNotFoundError(
                self._safe_message("Document URL returned 404 — resource not found")
            )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise CloudinaryDownloadError(
                self._safe_message(
                    f"Download failed with HTTP {exc.response.status_code}"
                )
            ) from exc

        return response.content

    # -- Validation helpers --

    def _validate_resource_type(self, resource_type: str) -> None:
        if resource_type not in SUPPORTED_RESOURCE_TYPES:
            raise CloudinaryUnsupportedTypeError(
                self._safe_message(
                    f"Resource type '{resource_type}' is not supported. "
                    f"Supported types: {', '.join(sorted(SUPPORTED_RESOURCE_TYPES))}"
                )
            )

    def _validate_format(self, file_format: str | None) -> None:
        if file_format is None:
            return
        if file_format.lower() not in SUPPORTED_DOCUMENT_FORMATS:
            raise CloudinaryUnsupportedFormatError(
                self._safe_message(
                    f"Format '{file_format}' is not supported. "
                    f"Supported formats: {', '.join(sorted(SUPPORTED_DOCUMENT_FORMATS))}"
                )
            )

    def _validate_size(self, size_bytes: int) -> None:
        if size_bytes > self._config.max_file_size_bytes:
            raise CloudinaryFileSizeLimitError(
                self._safe_message(
                    f"File size ({size_bytes} bytes) exceeds the maximum allowed "
                    f"({self._config.max_file_size_bytes} bytes)"
                )
            )

    # -- Format & filename helpers --

    @staticmethod
    def _extract_format(public_id: str) -> str | None:
        if "." in public_id:
            return public_id.rsplit(".", 1)[-1].lower()
        return None

    @staticmethod
    def _guess_format_from_url(url: str) -> str | None:
        path = url.split("?")[0]
        if "." in path:
            ext = path.rsplit(".", 1)[-1].lower()
            return ext
        return None

    @staticmethod
    def _build_filename(public_id: str, file_format: str | None = None) -> str:
        last_segment = public_id.rstrip("/").split("/")[-1]
        if "." not in last_segment and file_format:
            return f"{last_segment}.{file_format}"
        return last_segment

    @staticmethod
    def _guess_filename_from_url(url: str) -> str | None:
        path = url.split("?")[0]
        last_segment = path.rstrip("/").split("/")[-1]
        if last_segment and "." in last_segment:
            return last_segment
        return None

    # -- Metadata sanitisation --

    @staticmethod
    def _build_safe_metadata(api_resource: dict[str, Any]) -> dict[str, Any]:
        """Return only non-sensitive fields from the Admin API response."""
        safe_keys = {
            "public_id", "format", "version", "resource_type",
            "type", "created_at", "etag", "placeholder",
            "width", "height",
        }
        return {k: api_resource[k] for k in safe_keys if k in api_resource}
