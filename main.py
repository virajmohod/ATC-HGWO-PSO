"""
=========================================================
Main Entry Point
=========================================================
"""

from loguru import logger


def banner():

    print("=" * 60)

    print("ATC-HGWO-PSO")

    print("Adaptive Tent Chaotic Hybrid Grey Wolf")

    print("Particle Swarm Optimization")

    print("=" * 60)


def main():

    banner()

    logger.info("Research Framework Started")

    print()

    print("Project initialized successfully.")

    print()

    print("Next Step -> Data Loading")

    logger.success("Initialization Complete")


if __name__ == "__main__":

    main()