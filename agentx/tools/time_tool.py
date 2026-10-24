"""Current system time and timezone tool for AgentX.

Provides the host computer's local time, local timezone identifier,
UTC offset, UTC time, and epoch timestamp. Clearly distinguishes the
local timezone from UTC to provide grounded temporal context.
"""

from datetime import datetime, timezone
import time
from typing import Optional

from agentx.schemas import ToolResult, ToolStatus


def get_current_time(timezone_name: Optional[str] = None) -> ToolResult:
    """Retrieve the current local system time, timezone, and UTC time.

    Args:
        timezone_name: Optional timezone query or label requested by user.

    Returns:
        ToolResult with detailed local and UTC time evidence.
    """
    try:
        local_dt = datetime.now().astimezone()
        utc_dt = datetime.now(timezone.utc)

        local_tz = local_dt.tzname() or "Local Time"
        local_offset = local_dt.strftime("%z")
        if len(local_offset) == 5:
            local_offset_fmt = f"{local_offset[:3]}:{local_offset[3:]}"
        else:
            local_offset_fmt = local_offset or "+00:00"

        local_iso = local_dt.isoformat()
        local_human = local_dt.strftime("%Y-%m-%d %H:%M:%S")
        day_of_week = local_dt.strftime("%A")

        utc_iso = utc_dt.isoformat()
        utc_human = utc_dt.strftime("%Y-%m-%d %H:%M:%S")

        epoch_ts = time.time()

        message = (
            f"Current Local Time: {local_human} ({local_tz}, UTC{local_offset_fmt}, {day_of_week}) | "
            f"UTC Time: {utc_human} UTC"
        )

        data = {
            "local_time": local_human,
            "local_iso": local_iso,
            "local_timezone": local_tz,
            "local_utc_offset": f"UTC{local_offset_fmt}",
            "utc_time": utc_human,
            "utc_iso": utc_iso,
            "day_of_week": day_of_week,
            "epoch_timestamp": epoch_ts,
        }

        return ToolResult(
            status=ToolStatus.SUCCESS,
            data=data,
            message=message,
            evidence={
                "source": "system_clock",
                **data,
            },
        )
    except Exception as e:
        return ToolResult(
            status=ToolStatus.ERROR,
            error=f"Failed to retrieve system time: {str(e)}",
        )
