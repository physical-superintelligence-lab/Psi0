'''
For licensing see accompanying LICENSE.txt file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

List of skeletal transforms in the HDF5 files.
'''

# left fingers
LEFT_LITTLE = ['leftLittleFingerMetacarpal', 'leftLittleFingerKnuckle', 'leftLittleFingerIntermediateBase',  'leftLittleFingerIntermediateTip', 'leftLittleFingerTip']
LEFT_RING = [ 'leftRingFingerMetacarpal', 'leftRingFingerKnuckle', 'leftRingFingerIntermediateBase', 'leftRingFingerIntermediateTip', 'leftRingFingerTip']
LEFT_MIDDLE = ['leftMiddleFingerMetacarpal', 'leftMiddleFingerKnuckle', 'leftMiddleFingerIntermediateBase', 'leftMiddleFingerIntermediateTip', 'leftMiddleFingerTip']
LEFT_INDEX = ['leftIndexFingerMetacarpal', 'leftIndexFingerKnuckle', 'leftIndexFingerIntermediateBase', 'leftIndexFingerIntermediateTip', 'leftIndexFingerTip']
LEFT_THUMB = ['leftThumbKnuckle', 'leftThumbIntermediateBase', 'leftThumbIntermediateTip', 'leftThumbTip',]
LEFT_FINGERS = LEFT_LITTLE + LEFT_RING + LEFT_MIDDLE + LEFT_INDEX + LEFT_THUMB

# right fingers
RIGHT_LITTLE = ['rightLittleFingerMetacarpal', 'rightLittleFingerKnuckle', 'rightLittleFingerIntermediateBase',  'rightLittleFingerIntermediateTip', 'rightLittleFingerTip']
RIGHT_RING = [ 'rightRingFingerMetacarpal', 'rightRingFingerKnuckle', 'rightRingFingerIntermediateBase', 'rightRingFingerIntermediateTip', 'rightRingFingerTip']
RIGHT_MIDDLE = ['rightMiddleFingerMetacarpal', 'rightMiddleFingerKnuckle', 'rightMiddleFingerIntermediateBase', 'rightMiddleFingerIntermediateTip', 'rightMiddleFingerTip']
RIGHT_INDEX = ['rightIndexFingerMetacarpal', 'rightIndexFingerKnuckle', 'rightIndexFingerIntermediateBase', 'rightIndexFingerIntermediateTip', 'rightIndexFingerTip']
RIGHT_THUMB = ['rightThumbKnuckle', 'rightThumbIntermediateBase', 'rightThumbIntermediateTip', 'rightThumbTip',]
RIGHT_FINGERS = RIGHT_LITTLE + RIGHT_RING + RIGHT_MIDDLE + RIGHT_INDEX + RIGHT_THUMB

# left arm
LEFT_ARM = ['leftShoulder', 'leftArm', 'leftForearm', 'leftHand']

# right arm
RIGHT_ARM = ['rightShoulder', 'rightArm', 'rightForearm', 'rightHand']

# spine
SPINE = ['hip', 'spine1', 'spine2', 'spine3', 'spine4', 'spine5', 'spine6', 'spine7']

# neck
NECK = ['neck1', 'neck2', 'neck3', 'neck4']

# wrists
WRISTS = ['leftHand', 'rightHand']

DEFAULT_TFS = set(LEFT_FINGERS + LEFT_ARM + RIGHT_FINGERS + RIGHT_ARM + SPINE + NECK)


BODY_SKELETON_EDGES = [
    ('hip', 'spine1'), ('spine1', 'spine2'), ('spine2', 'spine3'), ('spine3', 'spine4'), ('spine4', 'spine5'), ('spine5', 'spine6'), ('spine6', 'spine7'), 
    ('spine7', 'neck1'), ('neck1', 'neck2'), ('neck2', 'neck3'), ('neck3', 'neck4'),
    ('spine7', 'leftShoulder'), ('leftShoulder', 'leftArm'), ('leftArm', 'leftForearm'), ('leftForearm', 'leftHand'),
    ('spine7', 'rightShoulder'), ('rightShoulder', 'rightArm'), ('rightArm', 'rightForearm'), ('rightForearm', 'rightHand'),
]