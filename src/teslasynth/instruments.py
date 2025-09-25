from dataclasses import dataclass


from .envelopes import ADSRConfig, CurveType
from .lfo import VibratoConfig


@dataclass(frozen=True)
class Instrument:
    envelope: ADSRConfig
    vibato: VibratoConfig | None = None


all: list[Instrument] = [
    # 0: Classic Piano - Quick attack, medium decay, moderate sustain, natural release
    Instrument(
        envelope=ADSRConfig(
            attack=5000,  # 5ms - fast attack
            decay=50000,  # 50ms - medium decay
            sustain_level=0.4,
            release=200000,  # 200ms - natural release
            curve=CurveType.Exponential,
        ),
        vibato=None,
    ),
    # 1: Electric Piano (Rhodes) - Slightly slower attack, longer decay
    Instrument(
        envelope=ADSRConfig(
            attack=15000,  # 15ms - percussive but not instant
            decay=80000,  # 80ms - bell-like decay
            sustain_level=0.6,
            release=300000,  # 300ms - smooth release
            curve=CurveType.Exponential,
        ),
        vibato=VibratoConfig(depth=0.02, freq=6.0),  # Subtle 6Hz vibrato
    ),
    # 2: Bright Synth Lead - Very fast attack, short decay, high sustain
    Instrument(
        envelope=ADSRConfig(
            attack=1000,  # 1ms - instant attack
            decay=20000,  # 20ms - quick decay to sustain
            sustain_level=0.8,
            release=100000,  # 100ms - snappy release
            curve=CurveType.Linear,  # Linear for punchy response
        ),
        vibato=VibratoConfig(depth=0.05, freq=5.5),  # Moderate vibrato
    ),
    # 3: Warm Pad - Slow attack, long decay, high sustain, long release
    Instrument(
        envelope=ADSRConfig(
            attack=100000,  # 100ms - slow attack swell
            decay=200000,  # 200ms - gradual decay
            sustain_level=0.9,
            release=1000000,  # 1s - long, smooth release
            curve=CurveType.Exponential,
        ),
        vibato=VibratoConfig(depth=0.03, freq=4.8),  # Gentle vibrato
    ),
    # 4: Aggressive Saw Lead - Instant attack, minimal decay, medium release
    Instrument(
        envelope=ADSRConfig(
            attack=500,  # 0.5ms - immediate response
            decay=10000,  # 10ms - barely any decay
            sustain_level=0.85,
            release=80000,  # 80ms - medium release
            curve=CurveType.Linear,
        ),
        vibato=VibratoConfig(depth=0.08, freq=7.0),  # Fast, pronounced vibrato
    ),
    # 5: Plucked String (Harp/Guitar) - Fast attack, rapid decay, short sustain
    Instrument(
        envelope=ADSRConfig(
            attack=2000,  # 2ms - quick pluck
            decay=30000,  # 30ms - fast decay
            sustain_level=0.2,
            release=50000,  # 50ms - short release
            curve=CurveType.Exponential,
        ),
        vibato=None,  # No vibrato for plucked sound
    ),
    # 6: Brass Section - Medium attack, controlled decay, medium sustain
    Instrument(
        envelope=ADSRConfig(
            attack=30000,  # 30ms - breath attack
            decay=60000,  # 60ms - controlled decay
            sustain_level=0.7,
            release=400000,  # 400ms - musical release
            curve=CurveType.Exponential,
        ),
        vibato=VibratoConfig(depth=0.04, freq=5.0),  # Musical vibrato
    ),
    # 7: Organ - Instant attack, no decay, instant release
    Instrument(
        envelope=ADSRConfig(
            attack=100,  # 0.1ms - immediate
            decay=0,  # No decay (stays at 1.0)
            sustain_level=1.0,
            release=5000,  # 5ms - quick cutoff
            curve=CurveType.Linear,
        ),
        vibato=None,  # Traditional organ has no vibrato
    ),
    # 8: Bell/Mallet - Instant attack, long decay, no sustain, long ring-out
    Instrument(
        envelope=ADSRConfig(
            attack=500,  # 0.5ms - immediate strike
            decay=800000,  # 800ms - long ringing decay
            sustain_level=0.0,  # No sustain level
            release=200000,  # 200ms - additional release
            curve=CurveType.Exponential,
        ),
        vibato=VibratoConfig(depth=0.01, freq=3.5),  # Subtle slow vibrato
    ),
    # 9: Bass Synth - Fast attack, quick decay, low sustain, short release
    Instrument(
        envelope=ADSRConfig(
            attack=2000,  # 2ms - punchy attack
            decay=15000,  # 15ms - tight decay
            sustain_level=0.3,
            release=30000,  # 30ms - quick release
            curve=CurveType.Linear,
        ),
        vibato=None,  # Clean bass, no vibrato
    ),
    # 10: Flute - Soft attack, minimal decay, long sustain, smooth release
    Instrument(
        envelope=ADSRConfig(
            attack=50000,  # 50ms - breathy attack
            decay=30000,  # 30ms - gentle shaping
            sustain_level=0.85,
            release=800000,  # 800ms - smooth, breathy release
            curve=CurveType.Exponential,
        ),
        vibato=VibratoConfig(depth=0.06, freq=5.5),  # Characteristic flute vibrato
    ),
    # 11: Oboe - Distinctive attack, controlled decay, medium sustain
    Instrument(
        envelope=ADSRConfig(
            attack=20000,  # 20ms - reed bite
            decay=40000,  # 40ms - nasal character
            sustain_level=0.75,
            release=600000,  # 600ms - reedy release
            curve=CurveType.Exponential,
        ),
        vibato=VibratoConfig(depth=0.07, freq=6.0),  # Prominent oboe vibrato
    ),
    # 12: Clarinet - Smooth attack, subtle decay, good sustain
    Instrument(
        envelope=ADSRConfig(
            attack=30000,  # 30ms - smooth reed attack
            decay=25000,  # 25ms - gentle shaping
            sustain_level=0.8,
            release=500000,  # 500ms - woody release
            curve=CurveType.Exponential,
        ),
        vibato=VibratoConfig(depth=0.04, freq=5.2),  # Subtle clarinet vibrato
    ),
    # 13: Saxophone (Alto) - Medium attack, breathy sustain, long release
    Instrument(
        envelope=ADSRConfig(
            attack=40000,  # 40ms - breathy reed attack
            decay=50000,  # 50ms - warm decay
            sustain_level=0.7,
            release=1200000,  # 1.2s - long, expressive release
            curve=CurveType.Exponential,
        ),
        vibato=VibratoConfig(depth=0.08, freq=4.5),  # Jazz vibrato
    ),
    # 14: Violin (Solo) - Slow attack, long decay, high sustain
    Instrument(
        envelope=ADSRConfig(
            attack=80000,  # 80ms - bow attack
            decay=150000,  # 150ms - gradual bow settling
            sustain_level=0.9,
            release=2000000,  # 2s - long bowed release
            curve=CurveType.Exponential,
        ),
        vibato=VibratoConfig(depth=0.10, freq=6.5),  # Expressive violin vibrato
    ),
    # 15: Cello - Slower attack, longer decay than violin
    Instrument(
        envelope=ADSRConfig(
            attack=120000,  # 120ms - deeper bow attack
            decay=250000,  # 250ms - rich decay
            sustain_level=0.85,
            release=2500000,  # 2.5s - very long release
            curve=CurveType.Exponential,
        ),
        vibato=VibratoConfig(depth=0.12, freq=5.8),  # Deep cello vibrato
    ),
    # 16: Acoustic Guitar (Fingerpicking) - Fast attack, medium decay
    Instrument(
        envelope=ADSRConfig(
            attack=3000,  # 3ms - string pluck
            decay=80000,  # 80ms - natural string decay
            sustain_level=0.3,
            release=100000,  # 100ms - quick release
            curve=CurveType.Exponential,
        ),
        vibato=None,  # No vibrato for fingerstyle
    ),
    # 17: Electric Guitar (Clean) - Medium attack, longer sustain
    Instrument(
        envelope=ADSRConfig(
            attack=8000,  # 8ms - pick attack
            decay=120000,  # 120ms - sustained tone
            sustain_level=0.6,
            release=300000,  # 300ms - medium release
            curve=CurveType.Linear,
        ),
        vibato=VibratoConfig(depth=0.03, freq=7.0),  # Subtle whammy
    ),
    # 18: Snare Drum - Instant attack, very fast decay
    Instrument(
        envelope=ADSRConfig(
            attack=100,  # 0.1ms - immediate strike
            decay=8000,  # 8ms - sharp decay
            sustain_level=0.0,
            release=2000,  # 2ms - quick cutoff
            curve=CurveType.Linear,
        ),
        vibato=None,
    ),
    # 19: Kick Drum - Instant attack, medium decay
    Instrument(
        envelope=ADSRConfig(
            attack=50,  # 0.05ms - immediate impact
            decay=40000,  # 40ms - punchy decay
            sustain_level=0.0,
            release=5000,  # 5ms - quick release
            curve=CurveType.Exponential,
        ),
        vibato=None,
    ),
    # 20: Analog Arpeggiator - Fast attack, minimal decay, short release
    Instrument(
        envelope=ADSRConfig(
            attack=2000,  # 2ms - quick response
            decay=15000,  # 15ms - slight shaping
            sustain_level=0.75,
            release=25000,  # 25ms - snappy release
            curve=CurveType.Linear,
        ),
        vibato=VibratoConfig(depth=0.02, freq=8.0),  # Fast, subtle vibrato
    ),
    # 21: Choir Voices - Very slow attack, long everything
    Instrument(
        envelope=ADSRConfig(
            attack=300000,  # 300ms - gradual vocal swell
            decay=500000,  # 500ms - slow shaping
            sustain_level=0.95,
            release=3000000,  # 3s - long vocal decay
            curve=CurveType.Exponential,
        ),
        vibato=VibratoConfig(depth=0.05, freq=4.0),  # Slow, choral vibrato
    ),
]

# Optional: Name mapping for easier reference
INSTRUMENT_NAMES = {
    0: "Classic Piano",
    1: "Electric Piano (Rhodes)",
    2: "Bright Synth Lead",
    3: "Warm Pad",
    4: "Aggressive Saw Lead",
    5: "Plucked String (Harp)",
    6: "Brass Section",
    7: "Hammond Organ",
    8: "Bell/Mallet",
    9: "Bass Synth",
    10: "Flute",
    11: "Oboe",
    12: "Clarinet",
    13: "Alto Saxophone",
    14: "Violin (Solo)",
    15: "Cello",
    16: "Acoustic Guitar (Finger)",
    17: "Electric Guitar (Clean)",
    18: "Snare Drum",
    19: "Kick Drum",
    20: "Analog Arpeggiator",
    21: "Choir Voices",
}
