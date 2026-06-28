import logging
import sys
from datetime import datetime, timedelta, timezone

from sunshine.bot import SunshineBot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

bot = SunshineBot()

since_time = datetime.now(timezone.utc) - timedelta(minutes=6)
count = bot.poll_once(since_time=since_time)
logging.info("Processed %d post(s)", count)

sys.exit(0)
