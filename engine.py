import asyncio
import os
import subprocess

import chess
import chess.engine
import chess.variant

from configs import EngineConfig, LimitConfig, SyzygyConfig


class Engine:
    def __init__(
        self,
        transport: asyncio.SubprocessTransport,
        engine: chess.engine.UciProtocol,
        ponder: bool,
        opponent: chess.engine.Opponent,
        limit_config: LimitConfig,
    ) -> None:
        self.transport = transport
        self.engine = engine
        self.ponder = ponder
        self.opponent = opponent
        self.limit_config = limit_config

    @classmethod
    async def from_config(
        cls, engine_config: EngineConfig, syzygy_config: SyzygyConfig, opponent: chess.engine.Opponent
    ) -> "Engine":
        stderr = subprocess.DEVNULL if engine_config.silence_stderr else None

        transport, engine = await chess.engine.popen_uci(engine_config.path, stderr=stderr)

        await cls._configure_engine(engine, engine_config, syzygy_config)
        await engine.send_opponent_information(opponent=opponent)

        return cls(transport, engine, engine_config.ponder, opponent, engine_config.limits)

    @classmethod
    async def test(cls, engine_config: EngineConfig) -> None:
        stderr = subprocess.DEVNULL if engine_config.silence_stderr else None

        transport, engine = await chess.engine.popen_uci(engine_config.path, stderr=stderr)
        await cls._configure_engine(engine, engine_config, SyzygyConfig(False, [], 0, False))
        board = cls._get_test_board(engine)
        # Движок может отвечать на uci и при этом зависать на поиске хода
        # (так вёл себя Crazyhouse-Stockfish). Без таймаута бот стоит на старте
        # бесконечно, поэтому ограничиваем проверку.
        try:
            result = await asyncio.wait_for(
                engine.play(board, chess.engine.Limit(time=0.1), info=chess.engine.INFO_ALL),
                timeout=60,
            )
        except asyncio.TimeoutError:
            await engine.quit()
            transport.close()
            raise RuntimeError(f'Engine "{engine_config.path}" did not answer a move within 60 s.') from None

        if not result.move:
            raise RuntimeError("Engine could not make a move!")

        await engine.quit()
        transport.close()

    @staticmethod
    def _get_test_board(engine: chess.engine.UciProtocol) -> chess.Board:
        # Dedicated variant engines (Atomic-Stockfish, Horde-Stockfish) refuse the
        # standard start position, so test them on a board they actually support.
        option = engine.options.get("UCI_Variant")
        variants = [v.lower() for v in (option.var if option is not None and option.var else [])]
        if variants and "chess" not in variants:
            board_class = {
                "atomic": chess.variant.AtomicBoard,
                "horde": chess.variant.HordeBoard,
                "antichess": chess.variant.AntichessBoard,
                "crazyhouse": chess.variant.CrazyhouseBoard,
                "kingofthehill": chess.variant.KingOfTheHillBoard,
                "3check": chess.variant.ThreeCheckBoard,
                "racingkings": chess.variant.RacingKingsBoard,
            }.get(variants[0])
            if board_class is not None:
                return board_class()
        return chess.Board()

    @staticmethod
    async def _configure_engine(
        engine: chess.engine.UciProtocol, engine_config: EngineConfig, syzygy_config: SyzygyConfig
    ) -> None:
        for name, value in engine_config.uci_options.items():
            if name.lower() in chess.engine.MANAGED_OPTIONS:
                print(f'UCI option "{name}" ignored as it is managed by the bot.')
            elif name in engine.options:
                await engine.configure({name: value})
            else:
                print(f'UCI option "{name}" ignored as it is not supported by the engine.')

        if not syzygy_config.enabled:
            return

        if "SyzygyPath" in engine.options and "SyzygyPath" not in engine_config.uci_options:
            delimiter = ";" if os.name == "nt" else ":"
            await engine.configure({"SyzygyPath": delimiter.join(syzygy_config.paths)})

        if "SyzygyProbeLimit" in engine.options and "SyzygyProbeLimit" not in engine_config.uci_options:
            await engine.configure({"SyzygyProbeLimit": syzygy_config.max_pieces})

    @property
    def name(self) -> str:
        return self.engine.id["name"]

    async def make_move(
        self, board: chess.Board, white_time: float, black_time: float, increment: float
    ) -> tuple[chess.Move, chess.engine.InfoDict]:
        if len(board.move_stack) < 2:
            time_limit = 10.0 if self.opponent.is_engine else 5.0
            if self.limit_config.time:
                time_limit = min(time_limit, self.limit_config.time)

            limit = chess.engine.Limit(time=time_limit, depth=self.limit_config.depth, nodes=self.limit_config.nodes)
            ponder = False
        else:
            limit = chess.engine.Limit(
                white_clock=white_time,
                white_inc=increment,
                black_clock=black_time,
                black_inc=increment,
                time=self.limit_config.time,
                depth=self.limit_config.depth,
                nodes=self.limit_config.nodes,
            )
            ponder = self.ponder

        low_time_limit = self._low_time_limit(board, white_time, black_time)
        if low_time_limit is not None:
            limit = low_time_limit
            ponder = False

        result = await self.engine.play(board, limit, info=chess.engine.INFO_ALL, ponder=ponder)

        if not result.move:
            raise RuntimeError("Engine could not make a move!")

        return result.move, result.info

    @staticmethod
    def _low_time_limit(
        board: chess.Board, white_time: float, black_time: float
    ) -> chess.engine.Limit | None:
        """Spend a fixed fraction of a short clock instead of trusting the engine clock.

        Engines keep thinking in proportion to the clock they see and know nothing about the
        network delay, so below `low_time_threshold` seconds the bot takes over and plays
        fast moves until the game ends or the clock recovers.
        """
        import extras

        settings = extras.section("clock")
        if not settings.get("enabled"):
            return None
        remaining = white_time if board.turn == chess.WHITE else black_time
        if remaining <= 0 or remaining > float(settings.get("low_time_threshold", 8.0)):
            return None
        horizon = max(1, int(settings.get("horizon", 30)))
        move_time = remaining / horizon
        move_time = max(move_time, float(settings.get("min_time", 0.04)))
        move_time = min(move_time, float(settings.get("max_time", 0.25)))
        move_time = max(0.015, min(move_time, remaining - 0.06))
        return chess.engine.Limit(time=move_time)

    async def start_pondering(self, board: chess.Board) -> None:
        if self.ponder:
            await self.engine.analysis(board)

    async def stop_pondering(self, board: chess.Board) -> None:
        if self.ponder:
            self.ponder = False
            await self.engine.analysis(board, chess.engine.Limit(time=0.001))

    async def close(self) -> None:
        try:
            await asyncio.wait_for(self.engine.quit(), 5.0)
        except TimeoutError:
            print("Engine could not be terminated cleanly.")

        self.transport.close()
