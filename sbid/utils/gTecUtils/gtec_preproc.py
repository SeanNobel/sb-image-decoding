import numpy as np
from nd.utils.gTecUtils import gTecUtils as gu


def eeg_subset_fromTrigger(args, raweeg_fname):
    d = gu.gTecDataset(raweeg_fname)
    eeg_chNum = 32

    # get Game Start/End indices
    GlobalT0_ind, GameStart_ind, GameEnd_ind = get_GameStartEnd_index(args, d)

    # print(f'game start: {GameStart_ind}')
    # print(f'game end: {GameEnd_ind}')

    raw = d.toMNE()
    eeg_time = raw.times

    eeg_trimmed = np.take(
        raw.get_data(), indices=np.arange(GameStart_ind, GameEnd_ind, 1), axis=1
    )
    eeg_trimmed = np.take(eeg_trimmed, indices=np.arange(0, eeg_chNum, 1), axis=0)

    eeg_time_trimmed = eeg_time[GameStart_ind:GameEnd_ind] - eeg_time[GlobalT0_ind]
    print(f"eeg_time_trimmed[:5] = {eeg_time_trimmed[:5]}")

    # correct eeg offset(for using g.tec interface of simultaneous EEG-Video recording)
    if "Video/VideoTimestamp" in d.info.keys():
        eeg_time_trimmed = eeg_time_trimmed - eeg_time_trimmed[0]

    return (
        eeg_trimmed,
        eeg_time_trimmed,
        (raw.ch_names[:eeg_chNum], raw.info["sfreq"], raw.get_montage()),
    )


def get_index(data, value):
    if value == 0:
        if (data[np.where(abs(data - value) == np.min(abs(data - value)))[0][0]]) < 0:
            return np.where(abs(data - value) == np.min(abs(data - value)))[0][0] + 1
        else:
            return np.where(abs(data - value) == np.min(abs(data - value)))[0][0]
    else:
        return np.where(abs(data - value) == np.min(abs(data - value)))[0][0]


def get_GameStartEnd_index(args, gtec_dataset):
    # for data doesn't have Trigger info
    if "Video/VideoTimestamp" in gtec_dataset.info.keys():
        # fps = 30  #video

        # offset/asynchrony info between EEG and video
        offsets = gtec_dataset.info["Video/VideoTimestamp"][0]

        eeg_offset = offsets[0] / 1e7  # unit: seconds
        video_offset = offsets[1] / args.video_fps  # unit: seconds
        print(f"eeg offset = {eeg_offset} seconds")
        print(f"video offset = {video_offset} seconds")
        print()

        # transform g.tec data into MNE format
        raw = gtec_dataset.toMNE()

        # correct EEG time offset
        eeg_time = (raw.times) - eeg_offset

        # find global Start and End in EEG
        eeg_global_StEnd_index = [get_index(eeg_time, 0), eeg_time.shape[0] - 1]

        return eeg_global_StEnd_index[0], eeg_global_StEnd_index[1]

    # data after 2021-12-20, having trigger info.
    if not ("Video/VideoTimestamp" in gtec_dataset.info.keys()):
        # sampling frequency
        sfreq = float(
            gtec_dataset.info["RawData/AcquisitionTaskDescription/SamplingFrequency"]
        )

        # two versions of the Trigger setting
        if (
            gtec_dataset.info["AsynchronData/AsynchronSignalTypes/Description"][0]
            == "Camera1_Hit"
        ):
            # v1. receive triggers from 5 cameras separately
            triggerName = gtec_dataset.info[
                "AsynchronData/AsynchronSignalTypes/Description"
            ]
        elif (
            gtec_dataset.info["AsynchronData/AsynchronSignalTypes/Description"][0]
            == "Hit"
        ):
            # v2. only one triger "CameraON"
            triggerName = [
                "Hit",
                "CameraON",
                "Self-report",
                "None",
                "None_",
                "Global_Unity_Start_End",
                "Lap",
            ]

        # pair triggerName and TriggerID
        triggerInfo = dict(
            zip(
                triggerName,  #
                gtec_dataset.info["AsynchronData/AsynchronSignalTypes/ID"],
            )
        )

        trigger_IDSeq = np.array(
            [np.array(xi[0]) for xi in gtec_dataset.info["AsynchronData/TypeID"]]
        )
        trigger_eegTimeIdxSeq = np.array(
            [np.array(xi[0]) for xi in gtec_dataset.info["AsynchronData/Time"]]
        )

        GlobalT0_index = trigger_eegTimeIdxSeq[
            np.where(trigger_IDSeq == int(triggerInfo["Global_Unity_Start_End"]))[0][0]
        ]
        if (
            len(
                np.where(trigger_IDSeq == int(triggerInfo["Global_Unity_Start_End"]))[0]
            )
            < 4
        ):
            # cond1: missing GameStart Trigger (Global_Unity_Start_End[1] > Lap[0])
            if (
                trigger_eegTimeIdxSeq[
                    np.where(
                        trigger_IDSeq == int(triggerInfo["Global_Unity_Start_End"])
                    )[0][1]
                ]
                > trigger_eegTimeIdxSeq[
                    np.where(trigger_IDSeq == int(triggerInfo["Lap"]))[0][0]
                ]
            ):
                game_StEnd_index = trigger_eegTimeIdxSeq[
                    np.where(trigger_IDSeq == int(triggerInfo["Lap"]))[0][0]
                ]
                game_StEnd_index = np.hstack(
                    (
                        game_StEnd_index,
                        trigger_eegTimeIdxSeq[
                            np.where(
                                trigger_IDSeq
                                == int(triggerInfo["Global_Unity_Start_End"])
                            )[0][-2]
                        ],
                    )
                )

            # cond2: missing GameEnd Trigger (Global_Unity_Start_End[-1] < Lap[-1])
            elif (
                trigger_eegTimeIdxSeq[
                    np.where(
                        trigger_IDSeq == int(triggerInfo["Global_Unity_Start_End"])
                    )[0][-1]
                ]
                < trigger_eegTimeIdxSeq[
                    np.where(trigger_IDSeq == int(triggerInfo["Lap"]))[0][-1]
                ]
            ):
                game_StEnd_index = trigger_eegTimeIdxSeq[
                    np.where(
                        trigger_IDSeq == int(triggerInfo["Global_Unity_Start_End"])
                    )[0][1]
                ]
                game_StEnd_index = np.hstack(
                    (
                        game_StEnd_index,
                        trigger_eegTimeIdxSeq[
                            np.where(trigger_IDSeq == int(triggerInfo["Lap"]))[0][-1]
                        ],
                    )
                )
        else:
            game_StEnd_index = trigger_eegTimeIdxSeq[
                np.where(trigger_IDSeq == int(triggerInfo["Global_Unity_Start_End"]))[
                    0
                ][1:3]
            ]

        return GlobalT0_index, game_StEnd_index[0], game_StEnd_index[1]
