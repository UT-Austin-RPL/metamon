from termcolor import colored


class ForwardException(Exception):
    def __init__(self, message: str):
        super().__init__(
            f"Handled Exception on Replay Forward-Fill: {colored(message, 'red')}"
        )


class ZoroarkException(ForwardException):
    def __init__(self):
        super().__init__(
            "Detected a Zoroark or related battle message (replace). The 3rd person replay format makes this difficult to deal with. Fix is TODO"
        )


class UnfinishedMessageException(ForwardException):
    def __init__(self, showdown_message: list[str]):
        super().__init__(
            f"This message: `{' '.join(showdown_message)}` is missing information. Check if this replay crashes on Showdown for the same reason. This tends to happen in extremely long battles (200+ turns) that are years old"
        )


class UnusualTeamSize(ForwardException):
    def __init__(self, size: int):
        super().__init__(f"Playing with {size} pokemon on a team.")


class MimicMiss(ForwardException):
    def __init__(self, message: str):
        super().__init__(f"Mimic logic failure: {message}")


class StrParsingException(ForwardException):
    def __init__(self, parse_func_name: str, inp: str):
        super().__init__(f"`{parse_func_name}` fails to parse message input: `{inp}`")


class SoftLockedGen(ForwardException):
    def __init__(self, gen: int):
        super().__init__(
            f"Triggered a branch that suggests Generation 5+ in a replay with a labeled generation of {gen}. The Replay Parser supports generations 1, 2, 3, and 4"
        )


class CantIDSwitchIn(ForwardException):
    def __init__(self, poke_details, poke_list):
        super().__init__(
            f"Could not ID switch `{poke_details}` in full team: `{poke_list}`"
        )


class UnhandledFromMoveItemLogic(ForwardException):
    def __init__(self, showdown_message: list[str]):
        super().__init__(
            f"Unhandled special case related to a move that swaps/adjusts items: `{' '.join(showdown_message)}`"
        )


class UnhandledFromOfAbilityLogic(ForwardException):
    def __init__(self, showdown_message: list[str]):
        super().__init__(
            f"Unhandled special case related to [from] [of] parsing in -ability: `{' '.join(showdown_message)}`"
        )


class UnhandledFromMoveItemLogic(ForwardException):
    def __init__(self, showdown_message: list[str]):
        super().__init__(
            f"Unhandled special case related to a move that swaps/adjusts items: `{' '.join(showdown_message)}`"
        )


class TrickError(ForwardException):
    def __init__(self, showdown_message: list[str]):
        super().__init__(
            f"Issue with Trick (move) logic: `{' '.join(showdown_message)}`"
        )


class CalledForeignConsecutive(ForwardException):
    def __init__(self, showdown_message: list[str]):
        super().__init__(
            f"Message `{' '.join(showdown_message)}` suggests a 'move that calls another move' decided to call a move that executes for multiple turns. We are very unlikely to block the discovery of subsequent uses of the copied move"
        )


class UnfinishedReplayException(ForwardException):
    def __init__(self, replay_url: str):
        super().__init__(
            f"Replay ended early or the link expired before the log could be scraped (link: {replay_url})"
        )


class IncompleteEffectLogic(ForwardException):
    def __init__(self, showdown_message: list[str]):
        super().__init__(
            f"Detected unimplemented [from] EFFECT/ABILITY/ITEM/MOVE [of] POKEMON simulator message: {' '.join(showdown_message)}"
        )


class UnimplementedSwapboost(ForwardException):
    def __init__(self, showdown_message: list[str]):
        super().__init__(
            f"`{' '.join(showdown_message)} triggers a -swapboost special case that is not implemented"
        )


class UnimplementedMoveFromMoveAbility(ForwardException):
    def __init__(self, showdown_message: list[str]):
        super().__init__(
            f"{' '.join(showdown_message)} led to a [from] effect [of] message in -move branch we could not make sense of"
        )


class UnimplementedMessage(ForwardException):
    def __init__(self, showdown_message: list[str]):
        super().__init__(
            f"Unimplemented (rare) showdown message `{' '.join(showdown_message)}`"
        )


class MoveInfoNotFound(ForwardException):
    def __init__(self, move_name: str):
        super().__init__(f"Move `{move_name}` not found in poke-env GenData info")


class ForwardVerify(ForwardException):
    def __init_(self, message: str):
        super().__init__(message)


class NoSpeciesClause(ForwardException):
    def __init__(self, replay):
        super().__init__(
            f"No 'Species Clause' Rule detected for replay: {replay.replay_url}"
        )


class Scalemons(ForwardException):
    def __init__(self, replay):
        super().__init__(f"{replay.replay_url} enables the Scalemons stats mod")


class RareValueError(ForwardException):
    def __init__(self, message: str):
        super().__init__(message)


class BackwardException(Exception):
    def __init__(self, message):
        super().__init__(
            f"Handled Exception on Replay Backward-Fill: {colored(message, 'blue')}"
        )


class TooFewMoves(BackwardException):
    def __init__(self, pokemon):
        super().__init__(
            f"Pokemon {pokemon} has too few moves (`had_moves = {pokemon.had_moves}"
        )


class TooManyMoves(BackwardException):
    def __init__(self, pokemon):
        super().__init__(
            f"Pokemon {pokemon} has too many moves (`moves = {pokemon.moves}"
        )


class ActionMisaligned(BackwardException):
    def __init__(self, pokemon, action):
        super().__init__(
            f"Assigning an action {action} that seems invalid for active pokemon {pokemon}"
        )


class ToNumpyError(BackwardException):
    def __init__(self, obs):
        super().__init__(
            f"Failed to convert observation to numpy array of expected size: {obs}"
        )


class PokedexMissingEntry(BackwardException):
    def __init__(self, raw_name: str, lookup_name: str):
        super().__init__(
            f"Pokemon '{raw_name}' (w/ lookup name '{lookup_name}') could not be found in the `poke_env` lookup data"
        )


class MovedexMissingEntry(BackwardException):
    def __init__(self, raw_name: str, lookup_name: str):
        super().__init__(
            f"Move '{raw_name}' (w/ lookup name '{lookup_name}') could not be found in the `poke_env` lookup data"
        )


class ForceSwitchMishandled(BackwardException):
    def __init__(self, subturn):
        super().__init__(
            f"A Subturn was reserved for a force switch but was never filled. Usually caused by undetectable switching move failure"
        )


class InvalidActionIndex(BackwardException):
    def __init__(self, state, action):
        super().__init__(f"Action `{action}` is not a valid choice in state `{state}`")


class InconsistentTeamPrediction(BackwardException):
    def __init__(self, team1, team2):
        super().__init__(
            f"Team `{team1}` is not consistent with predicted team `{team2}`"
        )
