from pathlib import Path

ROOT = Path("pwa")

folders = [

    # Backend
    "backend/app",
    "backend/app/api",
    "backend/app/api/v1",
    "backend/app/api/v1/endpoints",

    "backend/app/core",
    "backend/app/models",
    "backend/app/schemas",
    "backend/app/services",
    "backend/app/utils",
    "backend/app/middleware",
    "backend/app/config",

    # Frontend
    "frontend/public",
    "frontend/src",

    "frontend/src/assets",
    "frontend/src/components",
    "frontend/src/components/common",
    "frontend/src/components/layout",
    "frontend/src/components/scanner",

    "frontend/src/pages",

    "frontend/src/hooks",

    "frontend/src/services",

    "frontend/src/types",

    "frontend/src/utils",

    "frontend/src/styles",

    "frontend/src/context",

    "frontend/src/router",

    # Shared
    "shared",

    # Documentation
    "docs/api",
    "docs/architecture",
    "docs/user",

    # Docker
    "docker",

    # Scripts
    "scripts"
]

files = [

    # Backend

    "backend/requirements.txt",

    "backend/main.py",

    "backend/.env",

    "backend/.env.example",

    "backend/app/__init__.py",

    "backend/app/api/__init__.py",

    "backend/app/api/v1/__init__.py",

    "backend/app/api/v1/endpoints/__init__.py",

    "backend/app/api/v1/endpoints/health.py",

    "backend/app/api/v1/endpoints/scan.py",

    "backend/app/api/v1/endpoints/report.py",

    "backend/app/api/v1/endpoints/history.py",

    "backend/app/core/config.py",

    "backend/app/core/logger.py",

    "backend/app/core/security.py",

    "backend/app/services/qr_service.py",

    "backend/app/services/report_service.py",

    "backend/app/models/__init__.py",

    "backend/app/schemas/request.py",

    "backend/app/schemas/response.py",

    "backend/app/utils/helpers.py",

    "backend/app/middleware/cors.py",

    # Frontend

    "frontend/package.json",

    "frontend/vite.config.ts",

    "frontend/tsconfig.json",

    "frontend/index.html",

    "frontend/public/manifest.json",

    "frontend/public/favicon.ico",

    "frontend/public/robots.txt",

    "frontend/src/main.tsx",

    "frontend/src/App.tsx",

    "frontend/src/router/AppRouter.tsx",

    "frontend/src/pages/Home.tsx",

    "frontend/src/pages/Scanner.tsx",

    "frontend/src/pages/Upload.tsx",

    "frontend/src/pages/Results.tsx",

    "frontend/src/pages/History.tsx",

    "frontend/src/pages/Settings.tsx",

    "frontend/src/components/layout/Navbar.tsx",

    "frontend/src/components/layout/Footer.tsx",

    "frontend/src/components/scanner/CameraScanner.tsx",

    "frontend/src/components/scanner/ImageUploader.tsx",

    "frontend/src/components/scanner/BoundingBox.tsx",

    "frontend/src/components/common/Loader.tsx",

    "frontend/src/components/common/RiskBadge.tsx",

    "frontend/src/components/common/ErrorCard.tsx",

    "frontend/src/services/api.ts",

    "frontend/src/services/scanner.ts",

    "frontend/src/context/AppContext.tsx",

    "frontend/src/hooks/useCamera.ts",

    "frontend/src/hooks/useScanner.ts",

    "frontend/src/utils/constants.ts",

    "frontend/src/utils/helpers.ts",

    "frontend/src/types/api.ts",

    "frontend/src/styles/global.css",

    # Shared

    "shared/api_contract.md",

    "shared/openapi.yaml",

    # Docs

    "docs/api/API_REFERENCE.md",

    "docs/architecture/ARCHITECTURE.md",

    "docs/user/USER_GUIDE.md",

    # Docker

    "docker/docker-compose.yml",

    "docker/Dockerfile.backend",

    "docker/Dockerfile.frontend",

    # Scripts

    "scripts/run_backend.sh",

    "scripts/run_frontend.sh",

    "scripts/dev_setup.sh"
]

for folder in folders:
    Path(ROOT / folder).mkdir(parents=True, exist_ok=True)

for file in files:
    path = ROOT / file
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.touch()

print("QR Shield PWA structure created successfully.")