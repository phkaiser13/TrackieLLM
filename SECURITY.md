

## Security Policy — Trackie / Trackway

Trackie / Trackway is committed to keeping users, contributors and deployers safe. This document explains how to report security vulnerabilities, how we handle reports, and what to expect during coordinated disclosure.

> **Private reports only.** Do **not** open a public GitHub issue with exploit details. Public disclosure may put users at risk.

---

## How to report a vulnerability

Please report security issues **privately** using **one** of the methods below:

1. **Open a private GitHub issue** in this repository and mark the title with `[SECURITY]` (recommended):  
   `https://github.com/phkaiser13/TrackieLLM/issues`  
   — When creating the issue, choose the option to keep it confidential to maintain privacy.

2. **Email the project lead directly (confidential):**  
   **pedro.henrique@vytruve.org**

When possible, prefer the private GitHub issue route (it keeps vulnerability tracking adjacent to the codebase and allows secure attachments). If you email, please include sufficient detail and indicate whether you can accept encrypted messages.

---

## What to include in your report

To help us triage and fix the issue, include as much of the following information as you can safely share:

- A clear, concise summary of the issue and the impact.
- Versions affected (git commit SHA, release tag, package version, or build artifact).
- Reproduction steps (step-by-step) or a minimal proof-of-concept (PoC), ideally in a private attachment or private gist.
- Observed and expected behavior.
- Any logs, stack traces, or network captures that illustrate the issue.
- Network topology or deployment notes if relevant (on-device, cloud, mobile, docker).
- Whether the issue is likely to be exploitable remotely or requires local access.
- Your contact details and whether you consent to being credited publicly.
- Any suggested mitigation or workaround you’ve identified.

If you would like to send sensitive materials encrypted, indicate that in your initial message and we will provide a PGP key or alternative secure channel upon request.

---

## Scope — what's in-scope and out-of-scope

**In-scope**
- Vulnerabilities affecting official Trackie / Trackway repositories and released artifacts (code, model loaders, runtimes, server components).
- Issues in the provided model runners (ONNX / GGUF integration), LLM loaders, audio/vision pipelines when they result in security properties being violated.
- Deployment scripts, CI pipelines, or distribution artifacts published under the Trackie / Trackway project that could lead to privilege escalation, data leakage, or remote code execution.

**Out-of-scope**
- Vulnerabilities in third-party services not operated by the project (unless we ship an integration in a way that exposes user secrets).
- Physical device tampering or theft (we document mitigations, but hardware theft is outside the codebase scope).
- Social engineering directed against maintainers or users (report such incidents to maintainers and appropriate authorities).

If you are unsure whether an issue falls in scope, report it privately and we will triage.

---

## How we handle reports

When a private report is received we will:

1. **Acknowledge receipt** and create a confidential tracking ticket.
2. **Triage** the report to determine severity, likely exploitability, and affected components.
3. **Coordinate** with you on validation steps, additional data, and acceptable disclosure windows where relevant.
4. **Develop and test a fix** in a secure manner (private branches / patches).
5. **Publish a security advisory and release** with the fix and recommended mitigation once the patch is available and tested.
6. **Credit** contributors in the advisory if you consent to being credited.

We aim to be cooperative and transparent during this process. If you report a vulnerability in good faith, you will not face legal action from the Trackie / Trackway project for your actions in discovering or reporting the issue.

---

## Disclosure policy & timelines

We prefer **coordinated disclosure**: please do not publish exploit details publicly until a fix or mitigation is available and we have published an advisory. We will work with you to agree on a disclosure plan and to publish an advisory once users can safely upgrade.

(If you believe immediate public disclosure is necessary for user safety or other reasons, please indicate this in your report so we can discuss alternatives.)

---

## Mitigations & temporary workarounds

When applicable we will include in advisories:
- Short-term mitigations (configuration changes, disabling features).
- Compensating controls (network-level filtering, service isolation).
- Recommended upgrade or rollback instructions.

If you need urgent guidance for a deployed system (e.g., production device exposed to risk), mark the report as high-priority and include deployment details so we can provide targeted mitigations.

---

## Security testing & third-party audits

Trackie / Trackway conducts automated dependency scanning, static analysis and scheduled security reviews. We welcome responsible third-party audits—please contact us first so we can coordinate scope and avoid duplicate work.

If you are performing penetration testing or fuzzing against a hosted Trackie service or devices not under your control, obtain prior written permission from the maintainers.

---

## Credit and acknowledgement

We are grateful to security researchers who report issues responsibly. We will offer public acknowledgement in advisories when you consent. If you require anonymity, we will respect that.

---

## Contact & reporting summary

**Report a security issue (private):**
- Private GitHub issue (label `[SECURITY]`): `https://github.com/phkaiser13/TrackieLLM/issues`  
- Email (confidential): **pedro.henrique@vytruve.org**

If you need an encrypted channel or additional assurances for sensitive data transmission, state that in your initial email and we will respond with options for secure communication.

---

*Last updated: 2025-09-01*  
