"""Entry point of the `pipelines` container: registers every flow with Prefect and runs them."""

from prefect import flow, get_run_logger, serve

from abp_agent import __version__


@flow(name="platform-smoke-test", log_prints=True)
def smoke_test() -> str:
    """Trivial flow that proves the pipelines container can run work through Prefect."""
    get_run_logger().info("pipelines container is alive (abp-agent %s)", __version__)
    return "ok"


def main() -> None:
    serve(smoke_test.to_deployment(name="smoke-test", tags=["platform"]))


if __name__ == "__main__":
    main()
