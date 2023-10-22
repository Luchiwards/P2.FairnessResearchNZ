import argparse
from toolkit.analysis import plot_metrics, run_mitigations_justice, run_mitigations_police


def main():
    parser = argparse.ArgumentParser(description="Execute different Python scripts based on input parameters.")

    # Define the type parameter
    parser.add_argument("type", choices=["plot_metrics", "run_mitigations_justice", "run_mitigations_police"],
                        help="Type of operation to perform.")

    args = parser.parse_args()

    # Based on the argument, run the appropriate script
    if args.type == "plot_metrics":
        plot_metrics.plot()
    elif args.type == "run_mitigations_justice":
        run_mitigations_justice.run_mitigations()
    elif args.type == "run_mitigations_police":
        run_mitigations_police.run_mitigations()


if __name__ == "__main__":
    main()
