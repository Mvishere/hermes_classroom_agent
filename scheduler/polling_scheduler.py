"""APScheduler wrapper for periodic polling."""

import logging

from apscheduler.schedulers.background import BackgroundScheduler


class PollingScheduler:
    """Simple scheduler that runs a polling function on an interval."""

    def __init__(self, interval_minutes: int, job_func):
        self.interval_minutes = interval_minutes
        self.job_func = job_func
        self.scheduler = BackgroundScheduler(daemon=True)

    def start(self) -> None:
        logging.info("Starting scheduler with %s minute interval.", self.interval_minutes)
        self.scheduler.add_job(
            self.job_func,
            "interval",
            minutes=self.interval_minutes,
            id="polling_job",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=60,
        )
        self.scheduler.start()

    def shutdown(self) -> None:
        if self.scheduler.running:
            logging.info("Stopping scheduler.")
            self.scheduler.shutdown(wait=False)
