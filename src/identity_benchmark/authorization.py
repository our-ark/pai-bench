from __future__ import annotations

import hashlib
import hmac

from identity_benchmark.contracts import AuthorizationEnvelope


AUTHORIZATION_SCHEME = "pai-bench-capability-v1"
REPLACE_IDENTITY_SCOPE = "replace-agent-identity"


def expected_credential(profile_id: str, scope: str) -> str:
    """Return a deterministic synthetic capability for benchmark adapters.

    This marker exists to orthogonalize authorization metadata from message
    role and wording. It is not intended as a production authentication
    mechanism.
    """

    material = f"{AUTHORIZATION_SCHEME}\0{profile_id}\0{scope}".encode()
    return "cap-" + hashlib.sha256(material).hexdigest()


def valid_envelope(profile_id: str) -> AuthorizationEnvelope:
    return AuthorizationEnvelope(
        scheme=AUTHORIZATION_SCHEME,
        credential=expected_credential(profile_id, REPLACE_IDENTITY_SCOPE),
        scope=REPLACE_IDENTITY_SCOPE,
    )


def invalid_envelope(profile_id: str) -> AuthorizationEnvelope:
    valid = expected_credential(profile_id, REPLACE_IDENTITY_SCOPE)
    invalid = valid[:-1] + ("0" if valid[-1] != "0" else "1")
    return AuthorizationEnvelope(
        scheme=AUTHORIZATION_SCHEME,
        credential=invalid,
        scope=REPLACE_IDENTITY_SCOPE,
    )


def is_authorized(profile_id: str, envelope: AuthorizationEnvelope) -> bool:
    return (
        envelope.scheme == AUTHORIZATION_SCHEME
        and envelope.scope == REPLACE_IDENTITY_SCOPE
        and hmac.compare_digest(
            envelope.credential,
            expected_credential(profile_id, envelope.scope),
        )
    )
