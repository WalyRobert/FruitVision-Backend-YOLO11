# Security Policy

## Deployment status

FruitVision is an experimental backend. The current HTTP endpoints and WebSocket endpoint do not implement authentication or authorization. **Do not expose this service to the public internet or use it with private/user data until access control and resource protections are implemented and tested.**

## Current safeguards and limitations

- CORS is restricted to the local development origins by default. Set `FRUITVISION_CORS_ORIGINS` to a comma-separated allowlist of exact trusted frontend origins for a controlled deployment.
- CORS is not authentication. It does not prevent direct requests from scripts, servers, or command-line clients.
- Upload size/type validation, safe temporary-file handling, rate limiting, WebSocket access control, and request/resource quotas still require implementation and tests.
- Keep the service behind a trusted network boundary until those controls exist. Use HTTPS at the deployment boundary.

## Secrets and personal data

- Never commit API keys, passwords, tokens, private keys, credentials, or personal data.
- Do not upload private images/videos or personal data to an untrusted deployment.
- If a real credential is exposed, revoke/rotate it with its provider. Removing it from the latest commit does not remove it from Git history or existing clones.

## Reporting

Do not publish secrets or personal information in issues. Report security concerns privately to the repository owner.
