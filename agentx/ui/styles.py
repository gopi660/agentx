"""Textual CSS (TCSS) styling for AgentX Professional Terminal UI.

Implements an understated, restrained developer tool design system:
- Dark graphite / charcoal background (#121316, #18191c).
- Warm white primary text (#f4f4f5), muted gray secondary (#a1a1aa, #71717a).
- Desaturated sage green accent (#5ea28e) used selectively for active prompt and status.
- Thin subtle dividers (#27272a) instead of bright decorative borders.
- Dedicated, reliably visible bottom composer region with guaranteed height.
- Responsive collapsing of secondary telemetry on narrow terminals.
"""

AGENTX_TCSS = """
/* Global Screen Styling */
Screen {
    background: #121316;
    color: #f4f4f5;
    overflow: hidden;
    layout: vertical;
}

/* Header Area */
#app-header {
    dock: top;
    height: 2;
    background: #18191c;
    border-bottom: solid #27272a;
    padding: 0 1;
    layout: horizontal;
    content-align: left middle;
}

#header-title {
    width: auto;
    color: #f4f4f5;
    text-style: bold;
    padding-right: 1;
}

#header-divider-1 {
    width: 2;
    color: #3f3f46;
}

#header-status {
    width: auto;
    color: #5ea28e;
    padding-right: 1;
}

#header-divider-2 {
    width: 2;
    color: #3f3f46;
}

#header-session {
    width: 1fr;
    color: #71717a;
}

#header-help-hint {
    width: auto;
    color: #71717a;
    text-align: right;
}

/* Main Body Container */
#main-container {
    height: 1fr;
    layout: horizontal;
    background: #121316;
}

/* Left Column: Conversation */
#chat-container {
    width: 74fr;
    height: 100%;
    layout: vertical;
    background: #121316;
}

#conversation-history {
    height: 1fr;
    overflow-y: scroll;
    padding: 1 2;
    background: #121316;
}

/* Right Column: Telemetry & Activity */
#sidebar-container {
    width: 26fr;
    height: 100%;
    border-left: solid #27272a;
    layout: vertical;
    background: #151619;
}

/* Faux-3D Animation Widget (Subtle & Optional) */
#wireframe-panel {
    height: 7;
    background: #18191c;
    border-bottom: solid #27272a;
    margin: 0;
    padding: 0;
}

#wireframe-canvas {
    height: 100%;
    content-align: center middle;
    color: #5ea28e;
}

/* Live Activity Panel */
#activity-panel {
    height: 1fr;
    background: #151619;
    padding: 0;
    layout: vertical;
}

#activity-header {
    height: 1;
    background: #18191c;
    color: #a1a1aa;
    padding: 0 1;
    border-bottom: solid #27272a;
}

#activity-log {
    height: 1fr;
    overflow-y: scroll;
    padding: 0 1;
    color: #82828c;
}

/* System Metrics Bar */
#system-metrics {
    height: 5;
    border-top: solid #27272a;
    background: #18191c;
    padding: 0 1;
    color: #71717a;
}

/* Persistent Bottom Composer Area */
#composer-container {
    dock: bottom;
    height: 5;
    background: #18191c;
    border-top: solid #27272a;
    padding: 0 1;
    layout: vertical;
}

#prompt-input {
    width: 100%;
    height: 3;
    background: #1e1f23;
    border: tall #2e3036;
    color: #f4f4f5;
    padding: 0 1;
    margin: 0;
}

#prompt-input:focus {
    border: tall #5ea28e;
    background: #202227;
}

/* Keyboard Shortcut Hints */
#input-hints {
    height: 1;
    color: #71717a;
    content-align: left middle;
    padding: 0 1;
    margin: 0;
}

/* Conversation Message Elements */
.user-message-block {
    height: auto;
    margin: 1 0;
    padding: 0 1;
}

.user-message-header {
    height: auto;
    color: #5ea28e;
    text-style: bold;
    margin-bottom: 0;
}

.user-message-body {
    height: auto;
    color: #f4f4f5;
    padding-left: 2;
}

.agent-message-block {
    height: auto;
    margin: 1 0;
    padding: 0 1;
}

.agent-message-header {
    height: auto;
    color: #a1a1aa;
    text-style: bold;
    margin-bottom: 0;
}

.agent-message-body {
    height: auto;
    color: #f4f4f5;
    padding-left: 2;
}

.inline-tool-trace {
    height: auto;
    color: #71717a;
    padding-left: 2;
    margin: 0;
}

.system-notice-block {
    height: auto;
    color: #71717a;
    padding-left: 2;
    margin: 1 0;
}

.error-notice-block {
    height: auto;
    color: #f87171;
    padding-left: 2;
    margin: 1 0;
}

/* Startup Screen Styling */
#startup-dialog {
    width: 72;
    height: 22;
    background: #18191c;
    border: tall #27272a;
    padding: 1 2;
    align: center middle;
}

#startup-title {
    color: #f4f4f5;
    text-style: bold;
    margin-bottom: 1;
}

#startup-checks {
    height: 11;
    border: solid #27272a;
    background: #121316;
    padding: 1;
    color: #a1a1aa;
}

#startup-proceed-btn {
    margin-top: 1;
    background: #27272a;
    color: #f4f4f5;
    border: none;
}

#startup-proceed-btn:focus {
    background: #5ea28e;
    color: #121316;
}

/* Help Modal Styling */
#help-dialog {
    width: 66;
    height: 20;
    background: #18191c;
    border: tall #27272a;
    padding: 1 2;
    align: center middle;
}

#help-title {
    color: #f4f4f5;
    text-style: bold;
    margin-bottom: 1;
}

#help-content {
    height: 12;
    color: #a1a1aa;
    border: solid #27272a;
    background: #121316;
    padding: 1;
}
"""
