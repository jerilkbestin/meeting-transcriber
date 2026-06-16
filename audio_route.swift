import CoreAudio
import Foundation

let aggregateName = "Multi-Output Device"
let blackHoleName = "BlackHole 2ch"

struct AudioDeviceInfo {
    let id: AudioDeviceID
    let name: String
    let uid: String
    let inputChannels: Int
    let outputChannels: Int
    let classID: AudioClassID
}

enum RouteError: Error, CustomStringConvertible {
    case coreAudio(String, OSStatus)
    case missingDevice(String)
    case invalidSelection

    var description: String {
        switch self {
        case .coreAudio(let operation, let status):
            return "\(operation) failed with OSStatus \(status)"
        case .missingDevice(let name):
            return "Could not find required audio device: \(name)"
        case .invalidSelection:
            return "Invalid selection."
        }
    }
}

func propertyAddress(
    _ selector: AudioObjectPropertySelector,
    _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal
) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: scope,
        mElement: kAudioObjectPropertyElementMain
    )
}

func check(_ status: OSStatus, _ operation: String) throws {
    if status != noErr {
        throw RouteError.coreAudio(operation, status)
    }
}

func stringProperty(_ objectID: AudioObjectID, _ selector: AudioObjectPropertySelector) -> String? {
    var address = propertyAddress(selector)
    var value: Unmanaged<CFString>?
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    let status = AudioObjectGetPropertyData(objectID, &address, 0, nil, &size, &value)
    guard status == noErr, let value else {
        return nil
    }
    return value.takeUnretainedValue() as String
}

func uint32Property(_ objectID: AudioObjectID, _ selector: AudioObjectPropertySelector) -> UInt32? {
    var address = propertyAddress(selector)
    var value: UInt32 = 0
    var size = UInt32(MemoryLayout<UInt32>.size)
    let status = AudioObjectGetPropertyData(objectID, &address, 0, nil, &size, &value)
    guard status == noErr else {
        return nil
    }
    return value
}

func channelCount(_ deviceID: AudioDeviceID, scope: AudioObjectPropertyScope) -> Int {
    var address = propertyAddress(kAudioDevicePropertyStreamConfiguration, scope)
    var size: UInt32 = 0
    var status = AudioObjectGetPropertyDataSize(deviceID, &address, 0, nil, &size)
    guard status == noErr, size > 0 else {
        return 0
    }

    let raw = UnsafeMutableRawPointer.allocate(
        byteCount: Int(size),
        alignment: MemoryLayout<AudioBufferList>.alignment
    )
    defer { raw.deallocate() }

    status = AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, raw)
    guard status == noErr else {
        return 0
    }

    let list = UnsafeMutableAudioBufferListPointer(
        raw.bindMemory(to: AudioBufferList.self, capacity: 1)
    )
    return list.reduce(0) { $0 + Int($1.mNumberChannels) }
}

func allDevices() throws -> [AudioDeviceInfo] {
    var address = propertyAddress(kAudioHardwarePropertyDevices)
    var size: UInt32 = 0
    try check(
        AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size),
        "Reading device list size"
    )

    let count = Int(size) / MemoryLayout<AudioDeviceID>.size
    var ids = Array(repeating: AudioDeviceID(0), count: count)
    try check(
        AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &ids),
        "Reading device list"
    )

    return ids.compactMap { id in
        guard let name = stringProperty(id, kAudioObjectPropertyName),
              let uid = stringProperty(id, kAudioDevicePropertyDeviceUID),
              let classID = uint32Property(id, kAudioObjectPropertyClass) else {
            return nil
        }

        return AudioDeviceInfo(
            id: id,
            name: name,
            uid: uid,
            inputChannels: channelCount(id, scope: kAudioObjectPropertyScopeInput),
            outputChannels: channelCount(id, scope: kAudioObjectPropertyScopeOutput),
            classID: classID
        )
    }
}

func isSelectableOutput(_ device: AudioDeviceInfo) -> Bool {
    let lowered = device.name.lowercased()
    if device.outputChannels <= 0 {
        return false
    }
    if lowered.contains("blackhole") {
        return false
    }
    if lowered.contains("microsoft teams audio") {
        return false
    }
    if lowered == aggregateName.lowercased() {
        return false
    }
    if device.classID == kAudioAggregateDeviceClassID {
        return false
    }
    return true
}

func destroyExistingAggregate(named name: String, devices: [AudioDeviceInfo]) throws {
    let matching = devices.filter {
        $0.name == name && $0.classID == kAudioAggregateDeviceClassID
    }

    for device in matching {
        try check(
            AudioHardwareDestroyAggregateDevice(device.id),
            "Destroying existing \(name)"
        )
    }

    if !matching.isEmpty {
        Thread.sleep(forTimeInterval: 1.0)
    }
}

func createAggregate(selectedOutput: AudioDeviceInfo, blackHole: AudioDeviceInfo) throws -> AudioDeviceID {
    let subdevices: [[String: Any]] = [
        [
            kAudioSubDeviceUIDKey: selectedOutput.uid,
            kAudioSubDeviceDriftCompensationKey: 0
        ],
        [
            kAudioSubDeviceUIDKey: blackHole.uid,
            kAudioSubDeviceDriftCompensationKey: 1
        ]
    ]

    let description: [String: Any] = [
        kAudioAggregateDeviceNameKey: aggregateName,
        kAudioAggregateDeviceUIDKey: "com.local.meeting-transcriber.multi-output",
        kAudioAggregateDeviceSubDeviceListKey: subdevices,
        kAudioAggregateDeviceMainSubDeviceKey: selectedOutput.uid,
        kAudioAggregateDeviceClockDeviceKey: selectedOutput.uid,
        kAudioAggregateDeviceIsPrivateKey: false,
        kAudioAggregateDeviceIsStackedKey: true
    ]

    var lastStatus: OSStatus = noErr
    for attempt in 1...5 {
        var aggregateID = AudioDeviceID(0)
        lastStatus = AudioHardwareCreateAggregateDevice(description as CFDictionary, &aggregateID)
        if lastStatus == noErr {
            return aggregateID
        }
        if attempt < 5 {
            Thread.sleep(forTimeInterval: 0.5)
        }
    }

    throw RouteError.coreAudio("Creating \(aggregateName)", lastStatus)
}

func setDefaultOutput(_ deviceID: AudioDeviceID) {
    var mutableID = deviceID
    let size = UInt32(MemoryLayout<AudioDeviceID>.size)

    var defaultOutputAddress = propertyAddress(kAudioHardwarePropertyDefaultOutputDevice)
    let outputStatus = AudioObjectSetPropertyData(
        AudioObjectID(kAudioObjectSystemObject),
        &defaultOutputAddress,
        0,
        nil,
        size,
        &mutableID
    )

    var systemOutputAddress = propertyAddress(kAudioHardwarePropertyDefaultSystemOutputDevice)
    let systemStatus = AudioObjectSetPropertyData(
        AudioObjectID(kAudioObjectSystemObject),
        &systemOutputAddress,
        0,
        nil,
        size,
        &mutableID
    )

    if outputStatus != noErr {
        print("Warning: could not set default output device. OSStatus \(outputStatus)")
    }
    if systemStatus != noErr {
        print("Warning: could not set system output device. OSStatus \(systemStatus)")
    }
}

func promptForOutput(_ outputs: [AudioDeviceInfo]) throws -> AudioDeviceInfo {
    print("")
    print("Available physical output devices:")
    for (index, device) in outputs.enumerated() {
        print("  [\(index)] \(device.name) (\(device.outputChannels) out)")
    }
    print("")
    print("Select output device to pair with \(blackHoleName): ", terminator: "")

    guard let line = readLine(),
          let selected = Int(line),
          outputs.indices.contains(selected) else {
        throw RouteError.invalidSelection
    }
    return outputs[selected]
}

do {
    let devices = try allDevices()

    if devices.contains(where: {
        $0.name == aggregateName && $0.classID != kAudioAggregateDeviceClassID
    }) {
        throw RouteError.coreAudio(
            "\(aggregateName) exists but is not an aggregate device",
            kAudioHardwareBadDeviceError
        )
    }

    guard let blackHole = devices.first(where: {
        $0.name.lowercased().contains("blackhole") && $0.inputChannels > 0
    }) else {
        throw RouteError.missingDevice(blackHoleName)
    }

    let outputs = devices.filter(isSelectableOutput).sorted {
        $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending
    }

    guard !outputs.isEmpty else {
        throw RouteError.missingDevice("physical output device")
    }

    let selectedOutput = try promptForOutput(outputs)
    try destroyExistingAggregate(named: aggregateName, devices: devices)
    let aggregateID = try createAggregate(selectedOutput: selectedOutput, blackHole: blackHole)
    setDefaultOutput(aggregateID)

    print("")
    print("Configured \(aggregateName):")
    print("  Output : \(selectedOutput.name)")
    print("  Capture: \(blackHole.name)")
    print("")
    print("Set Microsoft Teams Speaker to '\(aggregateName)' if Teams does not follow the system output.")
} catch {
    fputs("Error: \(error)\n", stderr)
    exit(1)
}
