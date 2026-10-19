"""Relais TCP vers un serveur Ollama distant.

Utile quand le serveur Ollama (poste GPU, machine hors du réseau du
cluster...) n'est pas joignable directement : on expose un port local qui
relaie le trafic, et le cluster pointe sur ce port.

Exemple : relayer l'Ollama d'un poste GPU vers le réseau local.

    python scripts/dev/ollama_relay.py --listen 0.0.0.0:11435 --target 100.x.y.z:11434

Puis, côté déploiement : ``OLLAMA_HOST=http://<ip_de_cette_machine>:11435``.

Ce script est un outil de développement : il ne fait pas partie de l'API.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("ollama_relay")

BUFFER_SIZE = 64 * 1024


def parse_address(value: str) -> tuple[str, int]:
    """Découpe une adresse ``host:port``."""
    host, separator, port = value.rpartition(":")
    if not separator or not host or not port.isdigit():
        raise argparse.ArgumentTypeError(
            f"adresse attendue au format host:port, reçu {value!r}"
        )
    return host, int(port)


async def _pipe(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    """Recopie un flux jusqu'à la fermeture, dans un sens."""
    try:
        while chunk := await reader.read(BUFFER_SIZE):
            writer.write(chunk)
            await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        with contextlib.suppress(Exception):
            writer.close()
            await writer.wait_closed()


async def _handle(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    target: tuple[str, int],
) -> None:
    """Ouvre la connexion vers la cible et relaie dans les deux sens."""
    try:
        remote_reader, remote_writer = await asyncio.open_connection(*target)
    except OSError as exc:
        logger.warning("cible %s:%s injoignable : %s", target[0], target[1], exc)
        with contextlib.suppress(Exception):
            client_writer.close()
            await client_writer.wait_closed()
        return

    await asyncio.gather(
        _pipe(client_reader, remote_writer),
        _pipe(remote_reader, client_writer),
    )


async def serve(listen: tuple[str, int], target: tuple[str, int]) -> None:
    """Démarre le relais et sert indéfiniment."""
    server = await asyncio.start_server(
        lambda r, w: _handle(r, w, target),
        host=listen[0],
        port=listen[1],
    )
    addresses = ", ".join(str(sock.getsockname()[:2]) for sock in server.sockets)
    logger.info("relais actif sur %s -> %s:%s", addresses, *target)

    async with server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--listen",
        type=parse_address,
        default=parse_address("0.0.0.0:11435"),
        help="adresse d'écoute (défaut : 0.0.0.0:11435)",
    )
    parser.add_argument(
        "--target",
        type=parse_address,
        required=True,
        help="adresse du serveur Ollama à relayer (host:port)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(serve(args.listen, args.target))
    except KeyboardInterrupt:
        logger.info("relais arrêté")


if __name__ == "__main__":
    main()
