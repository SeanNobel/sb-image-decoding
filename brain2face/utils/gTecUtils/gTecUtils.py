import mne
import h5py
import numpy as np
import pandas as pd
from lxml import etree
import xml.etree.ElementTree as ET
from termcolor import cprint
from constants import MONTAGE_INFO_PATH

mne.set_log_level(verbose="WARNING")

# montageInfo_fpath = 'utils/gTecUtils/settings/montage_EEGonly_32ch.xml'


class gTecDataset:
    def __init__(self, filename, ch_range=None):
        self.filename = filename
        self.hdf5 = h5py.File(filename, "r")
        self.info = self.parser()
        self.data = self.info["RawData/Samples"].T * 1e-6  # to uV

        if ch_range == None:
            self.ch_range = (self.data).shape[0]

        else:
            self.ch_range = ch_range

        if "RawData/AcquisitionTaskDescription/ChannelProperties" in self.info:
            self.ch_names = list(
                self.info["RawData/AcquisitionTaskDescription/ChannelProperties"][
                    "ChannelName"
                ]
            )
            self.ch_types = list(
                self.info["RawData/AcquisitionTaskDescription/ChannelProperties"][
                    "ChannelType"
                ]
            )

            if type(self.ch_types[0]) is not str:
                self.loadChanInfo_standard()
        else:
            self.loadChanInfo_standard()

        # self.ch_types = ['eeg' for x in range(32)]
        if "RawData/AcquisitionTaskDescription/SamplingFrequency" in self.info:
            self.sfreq = int(
                self.info["RawData/AcquisitionTaskDescription/SamplingFrequency"]
            )
        else:
            v = getValueFromXML(
                self.hdf5["RawData"]["AcquisitionTaskDescription"][0], "SamplingFrequency"
            )
            self.sfreq = int(v)

    def parser(self):
        """Parse g.tec hdr5 format using h5py and lxml"""
        dataDict = {}

        def inner(name, obj):
            if type(obj) is h5py.Dataset:
                if "S" in str(obj.dtype):
                    if "xml" in str(obj[0][:20]):
                        try:
                            # raise Exception('xml error')
                            xmlObj = etree.fromstring(obj[0])
                            for key, value in xmlParser(xmlObj).items():
                                dataDict[name + "/" + key] = value
                        except Exception as ex:
                            print(
                                "warning: can not parse xml. Replace xml field to None."
                            )
                            print(ex)
                            dataDict[name] = None
                    else:
                        dataDict[name] = str(obj[0], "utf-8")
                else:
                    dataDict[name] = np.array(obj)

        self.hdf5.visititems(inner)
        return dataDict

    def loadChanInfo_standard(self, ch_names=True, ch_types=True):
        if ch_names:
            # fmt: off
            self.ch_names = [
                "FP1", "FP2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2",
                "FC6", "T7", "C3", "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3",
                "Pz", "P4", "P8", "PO7", "PO3", "PO4", "PO8", "Oz", "ACC_x", "ACC_y", "ACC_z",
                "BATTERY", "DigitalIN",
            ]
            # fmt: on

            self.ch_names = self.ch_names[: self.ch_range]

        if ch_types:
            ch_types = ["eeg" for x in range(32)]
            acc_bat = ["misc" for x in range(4)]
            ch_types.extend(acc_bat)
            ch_types.append("stim")

            self.ch_types = ch_types

            self.ch_types = self.ch_types[: self.ch_range]

    def toMNE(self):
        """Convert data into MNE RawArray format
        if ch_range is None, use 0~31 channels
        """

        # ch_names = [self.ch_names[v] for v in range(self.ch_range)]
        # ch_types = [self.ch_types[v] for v in range(self.ch_range)]
        data = self.data[range(self.ch_range), :]

        # print(self.ch_types)

        info = mne.create_info(
            ch_names=self.ch_names, sfreq=self.sfreq, ch_types=self.ch_types
        )
        raw = mne.io.RawArray(data, info)

        # load montage
        try:
            m = loadMontage(MONTAGE_INFO_PATH)
        except Exception as ex:
            print("warning: can not retrive montage. Use MNE standard_1020")
            print(ex)
            m = mne.channels.make_standard_montage("standard_1020")

        raw = raw.set_montage(m, match_case=False)
        return raw


def xmlParser(xmlObj):
    """Parse xml string using lxml. Convert content into string or pandas DataFrame"""
    xmlDict = {}

    def xmlParser_inner(xmlObj):
        # print(xmlObj)
        # list children
        if len(xmlObj) > 0:
            # print('#in have childern')
            childrenTag = [c.tag for c in xmlObj]

            if len(set(childrenTag)) > 1:
                # print('#in if different childern')
                # if different children
                for child in xmlObj:
                    xmlParser_inner(child)
            else:
                # print('#in if same childern')
                # if same children
                for child in xmlObj:
                    if len(child) > 1:
                        for child in xmlObj:
                            xmlParser_inner(child)
                    else:
                        df = pd.read_xml(etree.tostring(xmlObj))
                        xmlDict[xmlObj.tag] = df

        else:
            # print('#in no childern')
            # print(f':::  xmlObj.tag={xmlObj.tag}, xmlObj.text={xmlObj.text}')
            # if no child
            if xmlObj.tag in xmlDict.keys():
                if isinstance(xmlDict[xmlObj.tag], str):
                    if xmlObj.text != xmlDict[xmlObj.tag]:
                        xmlDict[xmlObj.tag] = list((xmlDict[xmlObj.tag], xmlObj.text))

                elif isinstance(xmlDict[xmlObj.tag], list):
                    if xmlObj.text not in xmlDict[xmlObj.tag]:
                        xmlDict[xmlObj.tag].append(xmlObj.text)
            else:
                xmlDict[xmlObj.tag] = xmlObj.text

        return xmlDict

    xmlParser_inner(xmlObj)
    return xmlDict


def getValueFromXML(xmlStr, tag):
    root = ET.fromstring(xmlStr)
    return root.find(tag).text


def montageParser(filename):
    """Parse montage xml file created from g.tec MontageCreator"""
    tree = etree.parse(filename)
    root = tree.getroot()

    d = dict()
    for c in root:
        v = c.text
        if type(v) is str:
            v = v.split(",")
            if len(v) > 1:
                if np.char.isnumeric(v[0]):
                    v = np.fromiter(v, dtype=np.float64)
            else:
                v = v[0]
        elif type(v) is type(None):
            v = "None"
        else:
            raise ValueError("can't recognize type")
        d[c.tag] = v

    return d


def loadMontage(fn: str):
    """load g.tec Montage file to MNE Montage class"""

    chanInfo = montageParser(fn)
    cXYZ = dict()
    for i, cName in enumerate(chanInfo["electrodename"]):
        xyz = np.array(
            [chanInfo["xposition"][i], chanInfo["yposition"][i], chanInfo["zposition"][i]]
        )
        cXYZ[cName] = xyz.astype(np.float32) / 1000

    return mne.channels.make_dig_montage(cXYZ)
