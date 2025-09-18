# Codex Addendum v2 — Dashboard v3 fixes (2025-09-07)

This document tracks frontend fixes made to the modern dashboard (dashboard_v3) and any concerns for follow‑up.

## Summary

- Fixed global search bar layout and layering on the Dashboard v3 header.
- Implemented a persistent, actionable microphone‑permission banner for Voice Note recording.
- Added permission re‑request and guidance flows; removed reliance on ephemeral toasts for critical permission issues.
- Prevented overlapping pop-ups by enforcing single-instance smart suggestions and limiting to one toast at a time.
- Added BottomBannerManager to serialize bottom banners (PWA prompt vs. mic-permission), ensuring only one bottom notification is visible at once.
  - Also auto-hides the top PWA banner while any bottom banner is visible and prevents showing the top banner if a bottom banner is currently shown.

## Changes

- File: `templates/dashboard_v3.html`
  - Header: added `sticky top-0 z-40` to keep the top bar anchored and above scrolled content.
  - Global search input: made width responsive (`w-full sm:w-80 md:w-96 lg:w-[28rem] max-w-[70vw]`) to prevent overflow when resizing and during scroll; retains existing look.
  - Added a new fixed, high‑z permission banner element `#microphonePermissionBanner` (bottom corner) with:
    - Clear message and tips
    - “Request Access” button that calls `navigator.mediaDevices.getUserMedia({audio:true})`
    - “How to allow” help action
    - Dismiss control (persists unless resolved or explicitly dismissed)
  - JavaScript helpers:
    - `checkMicrophonePermission()` uses `navigator.permissions.query({name:'microphone'})` when available; falls back gracefully.
    - `ensureMicrophonePermission()` shows the banner when state is `denied/prompt/unknown`.
    - `requestMicrophonePermission()` proactively requests access; on success hides banner and re‑initializes the recorder.
    - `showMicrophoneBanner()/hideMicrophoneBanner()/dismissMicrophoneBanner()` control visibility.
  - Recorder lifecycle:
    - On successful `initializeRecorder()`, the banner is hidden.
    - On error, the banner is shown with actionable guidance (replaces transient toast).
    - `showVoiceRecordingModal()` now calls `ensureMicrophonePermission()`.

## Rationale

- Search bar “sticking out” was primarily a stacking/overflow perception during scroll. Making the header sticky and giving the input a responsive max width eliminates overlapping/overflow artifacts while maintaining design intent.
- Permission prompts via toasts were easy to miss as the user scrolled. Mission‑critical permissions now use a persistent, fixed banner with a retry flow and short guidance that works across Chrome/Edge/Firefox/Safari.

## Concerns / Follow‑ups

- Safari support: `navigator.permissions` may not return a microphone state. The code treats this as `unknown` and still surfaces guidance + a manual request button.
- If a user hard‑blocks mic at the OS level, the banner remains until dismissed. Consider adding a “Don’t show again today” preference tied to `localStorage`.
- Accessibility: Banner is fixed and keyboard‑reachable; we can add `role="alert"` and focus management for even better a11y if desired.
- Tests: Consider adding a small UI test to assert the header has `position: sticky` and that `ensureMicrophonePermission()` shows the banner when `getUserMedia` throws.
- Toast UX: If you prefer stacked toasts, set `maxToasts` back >1 in `showToast`. For now we keep it to 1 to minimize simultaneous popups.

## Manual Test Checklist

- Scrolling dashboard: header remains in place; search input does not overlap cards or clip awkwardly at various widths.
- Resize window from mobile → desktop → ultra‑wide: input clamps to `max-w-[70vw]` and `lg:w-[28rem]`.
- Voice Note modal:
  - With permissions already granted: banner stays hidden; recording starts/stops normally.
  - With permissions denied: banner appears and persists; “Request Access” opens browser prompt when possible; guidance visible.
  - After granting: banner hides automatically; recorder re‑initializes; can record.

## Rollback

- File: `static/js/dashboard-help.js`
  - `showSmartSuggestion(...)` now enforces singleton behavior: removes any existing `.sb-smart-suggestion` before showing a new one. This prevents multiple bottom-left suggestion cards (e.g., daily reflection + tips) from appearing simultaneously.

- File: `templates/dashboard_v3.html` (toast system)
  - `showToast(...)` now enforces a maximum of one visible toast at a time by pruning older toasts from `#toastContainer` before adding a new one.

- Bottom banner serialization
  - Elements tagged with `data-bottom-banner` (`#microphonePermissionBanner`, `#pwaInstallPrompt`).
  - New helper `BottomBannerManager` with `show(name)/hide(name)/hideAll()`; used by mic banner helpers and wrapped around `showPWAInstallPrompt()` and `hidePWAInstallPrompt()`.
Revert the single file `templates/dashboard_v3.html` to the previous commit if needed.
