import logging
import sys

from sunshine.bot import SunshineBot
from sunshine.state import GCSStateStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

state = GCSStateStore()
bot = SunshineBot()

since_id = state.get_last_seen_post_id()
if since_id:
    bot._since_id = since_id
    logging.info("Resumed from post %s", since_id)
else:
    bot.bootstrap()
    logging.info("Bootstrapped — watching for posts newer than %s", bot._since_id)

count = bot.poll_once()
logging.info("Processed %d post(s)", count)

if bot._since_id:
    state.set_last_seen_post_id(bot._since_id)
    logging.info("Saved state: last_seen_post_id = %s", bot._since_id)

sys.exit(0)
