from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

FrameworkCategory = Literal[
    "framework",
    "orm",
    "template_engine",
    "testing",
    "library",
]


@dataclass
class FrameworkEntry:
    name: str
    category: FrameworkCategory
    dependency_packages: list[str] = field(default_factory=list)


_entries: list[FrameworkEntry] = []
_pkg_index: dict[str, list[FrameworkEntry]] = {}
_index_built: bool = False


def register(entry: FrameworkEntry) -> None:
    _entries.append(entry)


def all_entries() -> list[FrameworkEntry]:
    return list(_entries)


def _rebuild_index() -> None:
    global _index_built
    _pkg_index.clear()
    for e in _entries:
        for pkg in e.dependency_packages:
            _pkg_index.setdefault(pkg, []).append(e)
    _index_built = True


def match_by_dependency_package(package_name: str) -> list[FrameworkEntry]:
    if not _index_built:
        _rebuild_index()
    results: list[FrameworkEntry] = []
    seen: set[str] = set()
    pkg_lower = package_name.lower()
    for prefix, entries in _pkg_index.items():
        if pkg_lower == prefix or pkg_lower.startswith(prefix + ".") or pkg_lower.startswith(prefix + "/"):
            for e in entries:
                if e.name not in seen:
                    seen.add(e.name)
                    results.append(e)
    return results


# =========================================================================
# Framework data
# =========================================================================

# --- Python ---
register(FrameworkEntry(
    name="Django",
    category="framework",
    dependency_packages=["django"],
))
register(FrameworkEntry(
    name="Flask",
    category="framework",
    dependency_packages=["flask"],
))
register(FrameworkEntry(
    name="FastAPI",
    category="framework",
    dependency_packages=["fastapi"],
))
register(FrameworkEntry(
    name="Starlette",
    category="framework",
    dependency_packages=["starlette"],
))
register(FrameworkEntry(
    name="Tornado",
    category="framework",
    dependency_packages=["tornado"],
))
register(FrameworkEntry(
    name="Pyramid",
    category="framework",
    dependency_packages=["pyramid"],
))
register(FrameworkEntry(
    name="Bottle",
    category="framework",
    dependency_packages=["bottle"],
))
register(FrameworkEntry(
    name="Sanic",
    category="framework",
    dependency_packages=["sanic"],
))
register(FrameworkEntry(
    name="aiohttp",
    category="framework",
    dependency_packages=["aiohttp"],
))
register(FrameworkEntry(
    name="SQLAlchemy",
    category="orm",
    dependency_packages=["sqlalchemy"],
))
register(FrameworkEntry(
    name="Peewee",
    category="orm",
    dependency_packages=["peewee"],
))
register(FrameworkEntry(
    name="Tortoise ORM",
    category="orm",
    dependency_packages=["tortoise-orm"],
))
register(FrameworkEntry(
    name="Jinja2",
    category="template_engine",
    dependency_packages=["jinja2"],
))
register(FrameworkEntry(
    name="Mako",
    category="template_engine",
    dependency_packages=["mako"],
))
register(FrameworkEntry(
    name="Pytest",
    category="testing",
    dependency_packages=["pytest"],
))
register(FrameworkEntry(
    name="Celery",
    category="library",
    dependency_packages=["celery"],
))
register(FrameworkEntry(
    name="Pydantic",
    category="library",
    dependency_packages=["pydantic"],
))

# --- JavaScript / TypeScript ---
register(FrameworkEntry(
    name="React",
    category="framework",
    dependency_packages=["react"],
))
register(FrameworkEntry(
    name="Next.js",
    category="framework",
    dependency_packages=["next"],
))
register(FrameworkEntry(
    name="Vue",
    category="framework",
    dependency_packages=["vue"],
))
register(FrameworkEntry(
    name="Nuxt",
    category="framework",
    dependency_packages=["nuxt"],
))
register(FrameworkEntry(
    name="Angular",
    category="framework",
    dependency_packages=["@angular/core"],
))
register(FrameworkEntry(
    name="Svelte",
    category="framework",
    dependency_packages=["svelte"],
))
register(FrameworkEntry(
    name="SvelteKit",
    category="framework",
    dependency_packages=["@sveltejs/kit"],
))
register(FrameworkEntry(
    name="Express",
    category="framework",
    dependency_packages=["express"],
))
register(FrameworkEntry(
    name="Fastify",
    category="framework",
    dependency_packages=["fastify"],
))
register(FrameworkEntry(
    name="NestJS",
    category="framework",
    dependency_packages=["@nestjs/core"],
))
register(FrameworkEntry(
    name="Hono",
    category="framework",
    dependency_packages=["hono"],
))
register(FrameworkEntry(
    name="Solid",
    category="framework",
    dependency_packages=["solid-js"],
))
register(FrameworkEntry(
    name="Remix",
    category="framework",
    dependency_packages=["@remix-run/react"],
))
register(FrameworkEntry(
    name="Astro",
    category="framework",
    dependency_packages=["astro"],
))
register(FrameworkEntry(
    name="Koa",
    category="framework",
    dependency_packages=["koa"],
))
register(FrameworkEntry(
    name="Prisma",
    category="orm",
    dependency_packages=["@prisma/client"],
))
register(FrameworkEntry(
    name="TypeORM",
    category="orm",
    dependency_packages=["typeorm"],
))
register(FrameworkEntry(
    name="Drizzle ORM",
    category="orm",
    dependency_packages=["drizzle-orm"],
))
register(FrameworkEntry(
    name="Sequelize",
    category="orm",
    dependency_packages=["sequelize"],
))
register(FrameworkEntry(
    name="Mongoose",
    category="orm",
    dependency_packages=["mongoose"],
))
register(FrameworkEntry(
    name="Handlebars",
    category="template_engine",
    dependency_packages=["handlebars"],
))
register(FrameworkEntry(
    name="EJS",
    category="template_engine",
    dependency_packages=["ejs"],
))
register(FrameworkEntry(
    name="Pug",
    category="template_engine",
    dependency_packages=["pug"],
))
register(FrameworkEntry(
    name="Jest",
    category="testing",
    dependency_packages=["jest"],
))
register(FrameworkEntry(
    name="Vitest",
    category="testing",
    dependency_packages=["vitest"],
))
register(FrameworkEntry(
    name="Mocha",
    category="testing",
    dependency_packages=["mocha"],
))
register(FrameworkEntry(
    name="Cypress",
    category="testing",
    dependency_packages=["cypress"],
))
register(FrameworkEntry(
    name="Playwright",
    category="testing",
    dependency_packages=["@playwright/test"],
))
register(FrameworkEntry(
    name="Lodash",
    category="library",
    dependency_packages=["lodash"],
))
register(FrameworkEntry(
    name="Axios",
    category="library",
    dependency_packages=["axios"],
))
register(FrameworkEntry(
    name="Zod",
    category="library",
    dependency_packages=["zod"],
))
register(FrameworkEntry(
    name="tRPC",
    category="library",
    dependency_packages=["@trpc/server"],
))
register(FrameworkEntry(
    name="Redux",
    category="library",
    dependency_packages=["redux"],
))
register(FrameworkEntry(
    name="Zustand",
    category="library",
    dependency_packages=["zustand"],
))
register(FrameworkEntry(
    name="TanStack Query",
    category="library",
    dependency_packages=["@tanstack/react-query"],
))
register(FrameworkEntry(
    name="Tailwind CSS",
    category="library",
    dependency_packages=["tailwindcss"],
))

# --- PHP ---
register(FrameworkEntry(
    name="Laravel",
    category="framework",
    dependency_packages=["laravel/framework"],
))
register(FrameworkEntry(
    name="Symfony",
    category="framework",
    dependency_packages=["symfony/framework-bundle"],
))
register(FrameworkEntry(
    name="CodeIgniter",
    category="framework",
    dependency_packages=["codeigniter/framework"],
))
register(FrameworkEntry(
    name="CakePHP",
    category="framework",
    dependency_packages=["cakephp/cakephp"],
))
register(FrameworkEntry(
    name="Yii",
    category="framework",
    dependency_packages=["yiisoft/yii2"],
))
register(FrameworkEntry(
    name="Phalcon",
    category="framework",
    dependency_packages=["phalcon/cphalcon"],
))
register(FrameworkEntry(
    name="Slim",
    category="framework",
    dependency_packages=["slim/slim"],
))
register(FrameworkEntry(
    name="Blade",
    category="template_engine",
    dependency_packages=["illuminate/view"],
))
register(FrameworkEntry(
    name="Twig",
    category="template_engine",
    dependency_packages=["twig/twig"],
))
register(FrameworkEntry(
    name="Eloquent",
    category="orm",
    dependency_packages=["illuminate/database"],
))
register(FrameworkEntry(
    name="Doctrine",
    category="orm",
    dependency_packages=["doctrine/orm"],
))
register(FrameworkEntry(
    name="PHPUnit",
    category="testing",
    dependency_packages=["phpunit/phpunit"],
))
register(FrameworkEntry(
    name="Pest",
    category="testing",
    dependency_packages=["pestphp/pest"],
))
register(FrameworkEntry(
    name="Guzzle",
    category="library",
    dependency_packages=["guzzlehttp/guzzle"],
))
register(FrameworkEntry(
    name="Livewire",
    category="library",
    dependency_packages=["livewire/livewire"],
))
register(FrameworkEntry(
    name="Monolog",
    category="library",
    dependency_packages=["monolog/monolog"],
))

# --- Java ---
register(FrameworkEntry(
    name="Spring Boot",
    category="framework",
    dependency_packages=["spring-boot-starter", "spring-boot"],
))
register(FrameworkEntry(
    name="Spring Framework",
    category="framework",
    dependency_packages=["spring-core", "spring-context"],
))
register(FrameworkEntry(
    name="Jakarta EE",
    category="framework",
    dependency_packages=["jakarta.jakartaee"],
))
register(FrameworkEntry(
    name="Micronaut",
    category="framework",
    dependency_packages=["io.micronaut"],
))
register(FrameworkEntry(
    name="Quarkus",
    category="framework",
    dependency_packages=["io.quarkus"],
))
register(FrameworkEntry(
    name="Grails",
    category="framework",
    dependency_packages=["grails"],
))
register(FrameworkEntry(
    name="Play Framework",
    category="framework",
    dependency_packages=["play"],
))
register(FrameworkEntry(
    name="Hibernate",
    category="orm",
    dependency_packages=["hibernate-core"],
))
register(FrameworkEntry(
    name="MyBatis",
    category="orm",
    dependency_packages=["mybatis"],
))
register(FrameworkEntry(
    name="Thymeleaf",
    category="template_engine",
    dependency_packages=["thymeleaf"],
))
register(FrameworkEntry(
    name="JUnit",
    category="testing",
    dependency_packages=["junit", "junit-jupiter"],
))
register(FrameworkEntry(
    name="Mockito",
    category="testing",
    dependency_packages=["mockito-core"],
))
register(FrameworkEntry(
    name="TestNG",
    category="testing",
    dependency_packages=["testng"],
))
register(FrameworkEntry(
    name="Lombok",
    category="library",
    dependency_packages=["lombok"],
))

# --- C# / .NET ---
register(FrameworkEntry(
    name="ASP.NET Core",
    category="framework",
    dependency_packages=["microsoft.aspnetcore"],
))
register(FrameworkEntry(
    name="Blazor",
    category="framework",
    dependency_packages=["microsoft.aspnetcore.components"],
))
register(FrameworkEntry(
    name="Entity Framework Core",
    category="orm",
    dependency_packages=["microsoft.entityframeworkcore"],
))
register(FrameworkEntry(
    name="Dapper",
    category="orm",
    dependency_packages=["dapper"],
))
register(FrameworkEntry(
    name="NHibernate",
    category="orm",
    dependency_packages=["nhibernate"],
))
register(FrameworkEntry(
    name="xUnit",
    category="testing",
    dependency_packages=["xunit"],
))
register(FrameworkEntry(
    name="NUnit",
    category="testing",
    dependency_packages=["nunit"],
))
register(FrameworkEntry(
    name="Moq",
    category="testing",
    dependency_packages=["moq"],
))
register(FrameworkEntry(
    name="Serilog",
    category="library",
    dependency_packages=["serilog"],
))
register(FrameworkEntry(
    name="AutoMapper",
    category="library",
    dependency_packages=["automapper"],
))
register(FrameworkEntry(
    name="FluentValidation",
    category="library",
    dependency_packages=["fluentvalidation"],
))
register(FrameworkEntry(
    name="MediatR",
    category="library",
    dependency_packages=["mediatr"],
))

# --- Go ---
register(FrameworkEntry(
    name="Gin",
    category="framework",
    dependency_packages=["github.com/gin-gonic/gin"],
))
register(FrameworkEntry(
    name="Echo",
    category="framework",
    dependency_packages=["github.com/labstack/echo"],
))
register(FrameworkEntry(
    name="Fiber",
    category="framework",
    dependency_packages=["github.com/gofiber/fiber"],
))
register(FrameworkEntry(
    name="Chi",
    category="framework",
    dependency_packages=["github.com/go-chi/chi"],
))
register(FrameworkEntry(
    name="Gorilla Mux",
    category="framework",
    dependency_packages=["github.com/gorilla/mux"],
))
register(FrameworkEntry(
    name="Buffalo",
    category="framework",
    dependency_packages=["github.com/gobuffalo/buffalo"],
))
register(FrameworkEntry(
    name="Beego",
    category="framework",
    dependency_packages=["github.com/beego/beego"],
))
register(FrameworkEntry(
    name="GORM",
    category="orm",
    dependency_packages=["gorm.io/gorm"],
))
register(FrameworkEntry(
    name="Ent",
    category="orm",
    dependency_packages=["entgo.io/ent"],
))
register(FrameworkEntry(
    name="Testify",
    category="testing",
    dependency_packages=["github.com/stretchr/testify"],
))
register(FrameworkEntry(
    name="Cobra",
    category="library",
    dependency_packages=["github.com/spf13/cobra"],
))
register(FrameworkEntry(
    name="Viper",
    category="library",
    dependency_packages=["github.com/spf13/viper"],
))
register(FrameworkEntry(
    name="Zap",
    category="library",
    dependency_packages=["go.uber.org/zap"],
))
register(FrameworkEntry(
    name="Logrus",
    category="library",
    dependency_packages=["github.com/sirupsen/logrus"],
))

# --- Rust ---
register(FrameworkEntry(
    name="Actix-web",
    category="framework",
    dependency_packages=["actix-web"],
))
register(FrameworkEntry(
    name="Axum",
    category="framework",
    dependency_packages=["axum"],
))
register(FrameworkEntry(
    name="Rocket",
    category="framework",
    dependency_packages=["rocket"],
))
register(FrameworkEntry(
    name="Warp",
    category="framework",
    dependency_packages=["warp"],
))
register(FrameworkEntry(
    name="Tide",
    category="framework",
    dependency_packages=["tide"],
))
register(FrameworkEntry(
    name="Poem",
    category="framework",
    dependency_packages=["poem"],
))
register(FrameworkEntry(
    name="Diesel",
    category="orm",
    dependency_packages=["diesel"],
))
register(FrameworkEntry(
    name="SeaORM",
    category="orm",
    dependency_packages=["sea-orm"],
))
register(FrameworkEntry(
    name="Tokio",
    category="library",
    dependency_packages=["tokio"],
))
register(FrameworkEntry(
    name="Serde",
    category="library",
    dependency_packages=["serde"],
))
register(FrameworkEntry(
    name="Reqwest",
    category="library",
    dependency_packages=["reqwest"],
))
register(FrameworkEntry(
    name="Tracing",
    category="library",
    dependency_packages=["tracing"],
))
register(FrameworkEntry(
    name="Clap",
    category="library",
    dependency_packages=["clap"],
))

# --- Ruby ---
register(FrameworkEntry(
    name="Ruby on Rails",
    category="framework",
    dependency_packages=["rails"],
))
register(FrameworkEntry(
    name="Sinatra",
    category="framework",
    dependency_packages=["sinatra"],
))
register(FrameworkEntry(
    name="Hanami",
    category="framework",
    dependency_packages=["hanami"],
))
register(FrameworkEntry(
    name="Active Record",
    category="orm",
    dependency_packages=["activerecord"],
))
register(FrameworkEntry(
    name="Sequel",
    category="orm",
    dependency_packages=["sequel"],
))
register(FrameworkEntry(
    name="RSpec",
    category="testing",
    dependency_packages=["rspec-rails", "rspec"],
))
register(FrameworkEntry(
    name="Minitest",
    category="testing",
    dependency_packages=["minitest"],
))
register(FrameworkEntry(
    name="Capybara",
    category="testing",
    dependency_packages=["capybara"],
))
register(FrameworkEntry(
    name="Factory Bot",
    category="testing",
    dependency_packages=["factory_bot"],
))
register(FrameworkEntry(
    name="Devise",
    category="library",
    dependency_packages=["devise"],
))
register(FrameworkEntry(
    name="Sidekiq",
    category="library",
    dependency_packages=["sidekiq"],
))
register(FrameworkEntry(
    name="Kaminari",
    category="library",
    dependency_packages=["kaminari"],
))

# --- Swift ---
register(FrameworkEntry(
    name="SwiftUI",
    category="framework",
    dependency_packages=[],
))
register(FrameworkEntry(
    name="Vapor",
    category="framework",
    dependency_packages=["vapor"],
))
register(FrameworkEntry(
    name="Kitura",
    category="framework",
    dependency_packages=["kitura"],
))
register(FrameworkEntry(
    name="Alamofire",
    category="library",
    dependency_packages=["alamofire"],
))
register(FrameworkEntry(
    name="SnapKit",
    category="library",
    dependency_packages=["snapkit"],
))
register(FrameworkEntry(
    name="Kingfisher",
    category="library",
    dependency_packages=["kingfisher"],
))
register(FrameworkEntry(
    name="RxSwift",
    category="library",
    dependency_packages=["rxswift"],
))
register(FrameworkEntry(
    name="CoreData",
    category="orm",
    dependency_packages=[],
))
register(FrameworkEntry(
    name="SwiftData",
    category="orm",
    dependency_packages=[],
))
register(FrameworkEntry(
    name="XCTest",
    category="testing",
    dependency_packages=[],
))
register(FrameworkEntry(
    name="Quick",
    category="testing",
    dependency_packages=["quick"],
))
register(FrameworkEntry(
    name="Nimble",
    category="testing",
    dependency_packages=["nimble"],
))

# --- Flutter / Dart ---
register(FrameworkEntry(
    name="Flutter",
    category="framework",
    dependency_packages=["flutter"],
))
register(FrameworkEntry(
    name="Riverpod",
    category="library",
    dependency_packages=["riverpod"],
))
register(FrameworkEntry(
    name="Provider",
    category="library",
    dependency_packages=["provider"],
))
register(FrameworkEntry(
    name="Bloc",
    category="library",
    dependency_packages=["bloc"],
))
register(FrameworkEntry(
    name="GetX",
    category="library",
    dependency_packages=["get"],
))
register(FrameworkEntry(
    name="Dio",
    category="library",
    dependency_packages=["dio"],
))
register(FrameworkEntry(
    name="Hive",
    category="library",
    dependency_packages=["hive"],
))
