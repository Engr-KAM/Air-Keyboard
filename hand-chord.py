
import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
from pathlib import Path

# ==================================================
# CONFIGURATION
# ==================================================

MODEL_PATH = "models/hand_landmarker.task"
CAMERA_INDEX = 0

SAMPLE_RATE = 44100
DURATION = 3
STABLE_SECONDS = 0.4
VOLUME = 0.22

CHORDS = {
    1:  ("C MAJOR", [261.63, 329.63, 392.00]),
    2:  ("G MAJOR", [392.00, 493.88, 587.33]),
    3:  ("A MINOR", [440.00, 532.25, 659.25]),
    4:  ("F MAJOR", [349.23, 440.00, 523.25]),
    5:  ("D MINOR", [293.66, 349.32, 440.00]),
    6:  ("E MINOR", [329.63, 392.00, 493.88]),
    7:  ("D MAJOR", [293.66, 369.99, 440.00]),
    8:  ("E MAJOR", [329.63, 415.30, 493.88]),
    9:  ("B MINOR", [493.88, 587.33, 739.99]),
    10: ("A MAJOR", [440.00, 554.37, 659.25]),
}

# MediaPipe hand skeleton connections

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# Finger tip and PIP joint landmarks.
# These are the four non-thumb fingers.

FINGERS = [
    (8, 6),     # Index finger
    (12, 10),   # Middle finger
    (16, 14),   # Ring finger
    (20, 18),   # Little finger
]


# ==================================================
# AUDIO SETUP
# ==================================================

pygame.mixer.pre_init(
    frequency=SAMPLE_RATE,
    size=-16,
    channels=2,
    buffer=512
)

pygame.init()
pygame.mixer.set_num_channels(8)


def create_chord(frequencies):
    """Generate a short audio sample for a chord."""

    count = int(SAMPLE_RATE * DURATION)

    t = (
        np.arange(count, dtype=np.float32)
        / SAMPLE_RATE
    )

    wave = np.zeros(count, dtype=np.float32)

    for frequency in frequencies:
        # Fundamental frequency
        wave += np.sin(2 * np.pi * frequency * t)

        # A little second harmonic for a richer tone
        wave += 0.15 * np.sin(
            4 * np.pi * frequency * t
        )

    wave /= len(frequencies) * 1.15

    # Smooth beginning and ending to reduce clicks
    fade = int(SAMPLE_RATE * 0.03)

    envelope = np.ones(count, dtype=np.float32)

    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)

    wave *= envelope * VOLUME

    samples = np.int16(
        np.clip(wave, -1, 1) * 32767
    )

    # Convert mono samples to stereo
    stereo = np.column_stack((samples, samples))

    return pygame.sndarray.make_sound(
        np.ascontiguousarray(stereo)
    )


# Generate all chord sounds once at startup

SOUNDS = {
    number: create_chord(notes)
    for number, (_, notes) in CHORDS.items()
}


def play_chord(number):
    """Stop the previous sound and play the selected chord."""

    pygame.mixer.stop()

    SOUNDS[number].play()

    print(
        f"Playing {number} fingers: "
        f"{CHORDS[number][0]}"
    )


# ==================================================
# FINGER COUNTING
# ==================================================

def count_fingers(landmarks, handedness_label):
    """
    Count all five fingers on one hand.

    The four non-thumb fingers are checked by vertical
    position. The thumb is checked by horizontal position.

    The webcam frame is mirrored before MediaPipe processes
    it, so the handedness label is used with that convention.
    """

    count = 0

    # Count index, middle, ring and little fingers
    for tip, joint in FINGERS:
        if landmarks[tip].y < landmarks[joint].y:
            count += 1

    # Count the thumb separately
    thumb_tip = landmarks[4]
    thumb_joint = landmarks[3]

    if handedness_label == "Right":
        if thumb_tip.x > thumb_joint.x:
            count += 1

    elif handedness_label == "Left":
        if thumb_tip.x < thumb_joint.x:
            count += 1

    return count


# ==================================================
# DRAW HAND LANDMARKS
# ==================================================

def draw_hand(frame, landmarks):
    """Draw the detected hand skeleton on the webcam image."""

    height, width = frame.shape[:2]
    points = []

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    # Draw connections
    for start, end in CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 220, 100),
            2
        )

    # Draw landmark points
    for point in points:
        cv2.circle(
            frame,
            point,
            4,
            (0, 180, 255),
            -1
        )


# ==================================================
# MAIN PROGRAM
# ==================================================

def main():

    # Check that the MediaPipe model exists
    if not Path(MODEL_PATH).is_file():
        raise FileNotFoundError(
            f"Model missing: {MODEL_PATH}\n"
            "Place hand_landmarker.task in the models folder."
        )

    # Open webcam
    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError(
            "Cannot open webcam. Check webcam permissions "
            "or try CAMERA_INDEX = 1."
        )

    # MediaPipe Tasks API
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = (
        mp.tasks.vision.HandLandmarkerOptions
    )
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=RunningMode.VIDEO,

        # Detect both hands
        num_hands=2,

        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    # Gesture stability variables
    candidate = None
    candidate_since = time.monotonic()
    last_triggered = None

    start_time = time.monotonic()

    print("\n================================")
    print("10-FINGER HAND CHORD CONTROLLER")
    print("================================")
    print("1  = C Major")
    print("2  = G Major")
    print("3  = A Minor")
    print("4  = F Major")
    print("5  = D Minor")
    print("6  = E Minor")
    print("7  = D Major")
    print("8  = E Major")
    print("9  = B Minor")
    print("10 = A Major")
    print("--------------------------------")
    print("Both thumbs are included.")
    print("Use both hands for totals above 5.")
    print("Close all fingers to stop/reset.")
    print("Press Q to quit.")
    print("================================\n")

    try:
        with HandLandmarker.create_from_options(
            options
        ) as detector:

            while True:

                # Read webcam frame
                ok, frame = camera.read()

                if not ok:
                    print("Webcam frame unavailable.")
                    break

                # Mirror the image like a selfie camera
                frame = cv2.flip(frame, 1)

                # Convert BGR to RGB
                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb)
                )

                timestamp_ms = int(
                    (time.monotonic() - start_time) * 1000
                )

                # Detect hands
                result = detector.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                hands = result.hand_landmarks or []
                handedness_results = result.handedness or []

                total_fingers = 0
                left_count = 0
                right_count = 0

                # ------------------------------------------
                # COUNT FINGERS ON BOTH HANDS
                # ------------------------------------------

                for i, landmarks in enumerate(hands):

                    draw_hand(frame, landmarks)

                    hand_label = "Unknown"

                    # Get handedness classification
                    if i < len(handedness_results):
                        classifications = handedness_results[i]

                        if classifications:
                            category = classifications[0]

                            hand_label = (
                                getattr(
                                    category,
                                    "category_name",
                                    ""
                                )
                                or getattr(
                                    category,
                                    "display_name",
                                    ""
                                )
                            )

                    # Count this hand
                    hand_count = count_fingers(
                        landmarks,
                        hand_label
                    )

                    total_fingers += hand_count

                    if hand_label == "Left":
                        left_count = hand_count

                    elif hand_label == "Right":
                        right_count = hand_count

                # ------------------------------------------
                # SELECT GESTURE
                # ------------------------------------------

                gesture = None

                if not hands:
                    label = "Show your hand(s)"

                elif total_fingers == 0:
                    gesture = 0
                    label = "FIST - STOP / RESET"

                elif total_fingers in CHORDS:
                    gesture = total_fingers

                    chord_name = CHORDS[
                        total_fingers
                    ][0]

                    label = (
                        f"{total_fingers} FINGERS - "
                        f"{chord_name}"
                    )

                else:
                    label = "No matching chord"

                # ------------------------------------------
                # WAIT FOR A STABLE GESTURE
                # ------------------------------------------

                now = time.monotonic()

                if gesture != candidate:
                    candidate = gesture
                    candidate_since = now

                if now - candidate_since >= STABLE_SECONDS:

                    if candidate == 0:
                        # Stop sound while both hands are closed
                        pygame.mixer.stop()
                        last_triggered = None

                    elif candidate is None:
                        # Removing hands lets you select
                        # the same chord again next time
                        last_triggered = None

                    elif (
                        candidate in CHORDS
                        and candidate != last_triggered
                    ):
                        play_chord(candidate)
                        last_triggered = candidate

                # ------------------------------------------
                # DISPLAY ON SCREEN
                # ------------------------------------------

                cv2.putText(
                    frame,
                    label,
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (50, 230, 100),
                    2
                )

                selected = (
                    CHORDS[candidate][0]
                    if candidate in CHORDS
                    else "Waiting"
                )

                cv2.putText(
                    frame,
                    f"Selected: {selected}",
                    (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                finger_display = (
                    f"Total fingers: {total_fingers}/10"
                    if hands
                    else "Total fingers: --/10"
                )

                cv2.putText(
                    frame,
                    finger_display,
                    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Left hand: {left_count}",
                    (20, 165),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Right hand: {right_count}",
                    (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Hands detected: {len(hands)}",
                    (20, 235),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Q = quit",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.imshow(
                    "10-Finger Hand Chord Controller",
                    frame
                )

                # Quit with Q
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        pygame.mixer.stop()
        pygame.quit()


# ==================================================
# START PROGRAM
# ==================================================

if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print("ERROR:", error)

        pygame.quit()
