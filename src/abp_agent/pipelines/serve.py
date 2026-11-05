"""Entry point of the `pipelines` container: registers every flow with Prefect and runs them."""

from prefect import flow, get_run_logger, serve

from abp_agent import __version__
from abp_agent.pipelines.graph import build_graph
from abp_agent.pipelines.ingest import ingest


@flow(name="platform-smoke-test", log_prints=True)
def smoke_test() -> str:
    """Trivial flow that proves the pipelines container can run work through Prefect."""
    get_run_logger().info("pipelines container is alive (abp-agent %s)", __version__)
    return "ok"


def main() -> None:
    serve(
        smoke_test.to_deployment(name="smoke-test", tags=["platform"]),
        ingest.to_deployment(name="ingest", tags=["data"]),
        build_graph.to_deployment(name="build", tags=["data"]),
    )


if __name__ == "__main__":
    main()
