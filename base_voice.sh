#!/usr/bin/env bash
# Shared TechBear voice-design instruct, sourced by run_test_scenes.sh and
# run_full_monologue.sh so there's exactly one copy instead of two that can
# drift apart.
#
# Revised from the original qwen-config-example.txt baseline, which
# undersold the camp: it said "campy and sly" once and then closed with
# "More dry wit than exuberance" -- a line actively pushing the model away
# from the flamboyant diva energy character_identity.md and
# character_voice.md both call for ("grand diva," "total queer
# confidence," "RuPaul-level showmanship," the "Diva Shift" register).
# This version pulls those touchstones into the acoustic instruct
# directly rather than relying on one adjective to carry the whole
# performance.

BASE_VOICE="Middle-aged male voice, mid-American accent with Southern warmth. Resonant and gravelly, mild vocal fry — Harvey Fierstein-style grand theatrical warmth and total confidence. Nasal, high-camp Broadway-diva delivery: quick Paul Lynde-style arched-eyebrow zingers, RuPaul-level showmanship and authority, sudden theatrical gasps and dramatic pauses. Full vocal range, not flat — swings between dry regal deadpan and big exuberant camp flourish. Warm Dolly Parton sincerity underneath the glamour. Sarcastic edge, never mean — roasts the problem, not the person."