import logging

logging.basicConfig(
    filename='./logs.log',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    filemode='a'
    )
LOGGER = logging.getLogger(__name__)