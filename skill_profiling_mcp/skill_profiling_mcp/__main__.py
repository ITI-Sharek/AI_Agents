"""Independent entry point for the Skill Profiling MCP Server.

Start the server with:

    python -m skill_profiling_mcp

or, once installed:

    skill-profiling-mcp
"""

from skill_profiling_mcp.server import TRANSPORT, run


def main() -> None:
    run(transport=TRANSPORT)


if __name__ == "__main__":
    main()
