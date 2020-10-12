import logging

# ==============================================================================
# setup logger
# ==============================================================================


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)-12s %(levelname)-4s %(message)s")
logging.getLogger("mvae").setLevel(logging.INFO)
logger = logging.getLogger("mvae")
