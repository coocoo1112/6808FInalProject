//
//  RecordAudio.swift
//
//  This is a Swift class (updated for Swift 5)
//    that uses the iOS RemoteIO Audio Unit
//    to record audio input samples,
//  (should be instantiated as a singleton object.)
//
//  Created by Ronald Nicholson on 10/21/16.
//  Copyright © 2017,2019 HotPaw Productions. All rights reserved.
//  http://www.nicholson.com/rhn/
//  Distribution permission: BSD 2-clause license
//
//  Taken from https://gist.github.com/leonid-s-usov/dcd674b0a8baf96123cac6c4e08e3e0c

import Foundation
import AVFoundation
import AudioUnit
import Surge

struct FileStreamer {
    lazy var fileHandle = FileHandle(forWritingAtPath: logPath)

    lazy var logPath: String = {
        let path = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let logFilename = "\(Int(NSDate().timeIntervalSince1970)).csv"
        let filePath = path.appendingPathComponent(logFilename).absoluteString

        if FileManager.default.fileExists(atPath: filePath) == false {
            FileManager.default.createFile(atPath: filePath, contents: nil, attributes: nil)
        }
        print(filePath)
        return filePath
    }()

    mutating func write(_ string: String) {
        fileHandle?.seekToEndOfFile()
        if let data = string.data(using: String.Encoding.utf8){
            fileHandle?.write(data)
            print(data)
        }
    }

    mutating func close() {
        try! fileHandle?.close()
    }
}

class Logger {

    static var logFile: URL?
//    static var logFile: URL? {
//        guard let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return nil }
//        let formatter = DateFormatter()
//        formatter.dateFormat = "dd-MM-yyyy HH:mm:ss"
//        let dateString = formatter.string(from: Date())
//        let fileName = "\(dateString).csv"
//        return documentsDirectory.appendingPathComponent(fileName)
//    }

    static func setup() {
        guard let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
        let dateString = formatter.string(from: Date())
        let fileName = "\(dateString).csv"
        logFile = documentsDirectory.appendingPathComponent(fileName)
    }

    static func log(_ message: String) {
        guard let logFile = logFile else {
            return
        }

//        let formatter = DateFormatter()
//        formatter.dateFormat = "HH:mm:ss"
//        let timestamp = formatter.string(from: Date())
        //        guard let data = (timestamp + ": " + message + "\n").data(using: String.Encoding.utf8) else { return }
        // needs trailing comma
        guard let data = (message + ",").data(using: String.Encoding.utf8) else { return }

        if FileManager.default.fileExists(atPath: logFile.path) {
            if let fileHandle = try? FileHandle(forWritingTo: logFile) {
                fileHandle.seekToEndOfFile()
                fileHandle.write(data)
                fileHandle.closeFile()
            }
        } else {
            try? data.write(to: logFile, options: .atomicWrite)
        }
    }
}

// call setupAudioSessionForRecording() during controlling view load
// call startRecording() to start recording in a later UI call
var gTmp0 = 0

final class RecordAudio: NSObject, ObservableObject {

    var auAudioUnit: AUAudioUnit! = nil

    var enableRecording     = true
    var audioSessionActive  = false
    var audioSetupComplete  = false
    var isRecording         = false

    var isPlaying           = false

    // HACK so we don't run the FFT too often
    var cycleCount          = 0

    var sampleRate: Double  =  48000.0      // desired audio sample rate

    let circBuffSize        =  32768        // lock-free circular fifo/buffer size
    var circBuffer          = [Float](repeating: 0, count: 32768)
    // HACK need to make sure this runs on the main UI thread somehow
    // not sure if this is what you're supposed to do
//    @Published var publishedCircBuffer          = [Float](repeating: 0, count: 32768)
    let fftPlotSize         = 256
    @Published var fftPlot  = [Float](repeating: 0, count: 256)
    var circInIdx: Int      =  0            // sample input  index
    var circOutIdx: Int     =  0            // sample output index

    var audioLevel: Float   = 0.0

    private var micPermissionRequested  = false
    private var micPermissionGranted    = false

    // for restart from audio interruption notification
    private var audioInterrupted        = false

    private var renderBlock: AURenderBlock?

//    var fileURL: URL? = nil
    var fileStreamer = FileStreamer()
//    var fileOutputStream: TextOutputStream? = nil

    func startPlayback(soundFileURL: URL) {
        if isPlaying { return }

        if audioSessionActive == false {
            // configure and activate Audio Session, this might change the sampleRate
            setupAudioSessionForRecording()
        }

        do {
            player = try AVAudioPlayer(
                contentsOf: soundFileURL
            )

            player?.play()
        } catch {   // error handling placeholder
            return
        }
    }

    func startRecording() {

        if isRecording { return }

        if audioSessionActive == false {
            // configure and activate Audio Session, this might change the sampleRate
            setupAudioSessionForRecording()
        }

        guard micPermissionGranted && audioSessionActive else { return }

        let audioFormat = AVAudioFormat(
            commonFormat: AVAudioCommonFormat.pcmFormatFloat32,   // pcmFormatInt16, pcmFormatFloat32,
            sampleRate: Double(sampleRate),                     // 44100.0 48000.0
            channels: 1,                                         // 1 or 2
            interleaved: true )                                 // true for interleaved stereo

        if auAudioUnit == nil {
            setupRemoteIOAudioUnitForRecord(audioFormat: audioFormat!)
        }

        renderBlock = auAudioUnit.renderBlock  //  returns AURenderBlock()

        if    enableRecording
            && micPermissionGranted
            && audioSetupComplete
            && audioSessionActive
            && isRecording == false {

            Logger.setup()

            auAudioUnit.inputHandler = { (actionFlags, timestamp, frameCount, inputBusNumber) in
                if let block = self.renderBlock {       // AURenderBlock?
                    var bufferList = AudioBufferList(
                        mNumberBuffers: 1,
                        mBuffers: AudioBuffer(
                            mNumberChannels: audioFormat!.channelCount,
                            mDataByteSize: 0,
                            mData: nil))

                    let err: OSStatus = block(actionFlags,
                                               timestamp,
                                               frameCount,
                                               inputBusNumber,
                                               &bufferList,
                                               .none)
                    if err == noErr {
                        // save samples from current input buffer to circular buffer
                        self.recordMicrophoneInputSamples(
                            inputDataList: &bufferList,
                            frameCount: UInt32(frameCount) )
                    }
                }
            }

            auAudioUnit.isInputEnabled  = true

            do {
                circInIdx   =   0                       // initialize circular buffer pointers
                circOutIdx  =   0
                try auAudioUnit.allocateRenderResources()
                try auAudioUnit.startHardware()         // equivalent to AudioOutputUnitStart ???
                isRecording = true

            } catch let e {
                print(e)
            }
        }
    }

    func stopRecording() {

//        fileOutputStream?.close()
//        fileStreamer.close()
        Logger.setup()
        if isRecording {
            auAudioUnit.stopHardware()
            isRecording = false
        }
        if audioSessionActive {
            let audioSession = AVAudioSession.sharedInstance()
            do {
                try audioSession.setActive(false)
            } catch /* let error as NSError */ {
            }
            audioSessionActive = false
        }
    }

    private func recordMicrophoneInputSamples(   // process RemoteIO Buffer from mic input
        inputDataList: UnsafeMutablePointer<AudioBufferList>,
        frameCount: UInt32 ) {
        let inputDataPtr = UnsafeMutableAudioBufferListPointer(inputDataList)
        let mBuffers: AudioBuffer = inputDataPtr[0]

        // Microphone Input Analysis
        // let data      = UnsafePointer<Int16>(mBuffers.mData)
        let bufferPointer = UnsafeMutableRawPointer(mBuffers.mData)
        if let bptr = bufferPointer {
            let dataArray = bptr.assumingMemoryBound(to: Float32.self)
            var sum: Float32 = 0.0
            var j = self.circInIdx
            let m = self.circBuffSize

            // TODO this doesn't actually handle more than one channel,
            // which is all the iPad has
            let floatArray = Array(UnsafeBufferPointer(start: dataArray, count: Int(frameCount)))
            let stringDataArray = floatArray.map{ String($0) }
            let stringData = stringDataArray.joined(separator: ",")
//            fileStreamer.write(stringData)
            Logger.log(stringData)
//            if fileURL != nil {
//                try! fileStreamer.write(to: fileURL!, atomically: true, encoding: .utf8)
//
//                print("writing")
//                print(fileURL)
//            }

            for i in 0..<Int(frameCount/mBuffers.mNumberChannels) {
                for ch in 0..<Int(mBuffers.mNumberChannels) {
                    let x = Float32(dataArray[i+ch])   // copy channel sample
                    self.circBuffer[j+ch] = x
                    sum += x*x
                }

                j += Int(mBuffers.mNumberChannels)
                if j >= m { j = 0 }                // into circular buffer
            }
            self.circInIdx = j              // circular index will always be less than size
            // measuredMicVol_1 = sqrt( Float(sum) / Float(count) ) // scaled volume
            if sum > 0.0 && frameCount > 0 {
                let tmp = 5.0 * (logf(sum / Float32(frameCount)) + 20.0)
                let r: Float32 = 0.2
                audioLevel = r * tmp + (1.0 - r) * audioLevel
            }
        }

        // HACK fft takes too long, so make sure we don't call it too often
        // otherwise the UI seems to slow down
        /*
        let CYCLES_PER_FFT = 16
        if cycleCount % CYCLES_PER_FFT == 0 {
            DispatchQueue.global().async { [self] in
                let fft = Surge.fft(circBuffer)

                // reduce FFT size, since we don't want to plot tens of thousands of points
                let kernelSize = circBuffSize / fftPlotSize
                var newFftPlot = [Float](repeating: 0, count: fftPlotSize)

                for i in 0..<fftPlotSize {
    //                newFftPlot[i] = Surge.sum(fft[i * kernelSize..<(i + 1) * kernelSize])
                    // actually, the FFT is mirrored (correctly?)
                    // https://dsp.stackexchange.com/questions/4825/why-is-the-fft-mirrored
                    // so we only need to take half of it
                    newFftPlot[i] = Surge.sum(fft[i * kernelSize / 2..<(i + 1) * kernelSize / 2])
                }

                // update UI thread
                DispatchQueue.main.async { [self] in
                    fftPlot = newFftPlot
                }
            }
        }
        */

        cycleCount += 1
    }

    // set up and activate Audio Session
    func setupAudioSessionForRecording() {
        do {

            let audioSession = AVAudioSession.sharedInstance()

            if micPermissionGranted == false {
                if micPermissionRequested == false {
                    micPermissionRequested = true
                    audioSession.requestRecordPermission({(granted: Bool) -> Void in
                        if granted {
                            self.micPermissionGranted = true
                            self.startRecording()
                            return
                        } else {
                            self.enableRecording = false
                            // dispatch in main/UI thread an alert
                            //   informing that mic permission is not switched on
                        }
                    })
                }
                return
            }

            if enableRecording {
                try audioSession.setCategory(AVAudioSession.Category.playAndRecord)
                try audioSession.setMode(.measurement)
            }
            let preferredIOBufferDuration = 0.0053  // 5.3 milliseconds = 256 samples
            try audioSession.setPreferredSampleRate(sampleRate) // at 48000.0
            try audioSession.setPreferredIOBufferDuration(preferredIOBufferDuration)

            NotificationCenter.default.addObserver(
                forName: AVAudioSession.interruptionNotification,
                object: nil,
                queue: nil,
                using: myAudioSessionInterruptionHandler )

            try audioSession.setActive(true)
            audioSessionActive = true
            self.sampleRate = audioSession.sampleRate
        } catch /* let error as NSError */ {
            // placeholder for error handling
        }
    }

    // find and set up the sample format for the RemoteIO Audio Unit
    private func setupRemoteIOAudioUnitForRecord(audioFormat: AVAudioFormat) {

        do {
            let audioComponentDescription = AudioComponentDescription(
                componentType: kAudioUnitType_Output,
                componentSubType: kAudioUnitSubType_RemoteIO,
                componentManufacturer: kAudioUnitManufacturer_Apple,
                componentFlags: 0,
                componentFlagsMask: 0 )

            try auAudioUnit = AUAudioUnit(componentDescription: audioComponentDescription)

            // bus 1 is for data that the microphone exports out to the handler block
            let bus1 = auAudioUnit.outputBusses[1]

            try bus1.setFormat(audioFormat)  //      for microphone bus
            audioSetupComplete = true
        } catch let error {
            print(error)
        }
    }

    private func myAudioSessionInterruptionHandler(notification: Notification) {
        let interuptionDict = notification.userInfo
        if let interuptionType = interuptionDict?[AVAudioSessionInterruptionTypeKey] {
            let interuptionVal = AVAudioSession.InterruptionType(
                rawValue: (interuptionType as AnyObject).uintValue )
            if interuptionVal == AVAudioSession.InterruptionType.began {
                // [self beginInterruption];
                if isRecording {
                    auAudioUnit.stopHardware()
                    isRecording = false
                    let audioSession = AVAudioSession.sharedInstance()
                    do {
                        try audioSession.setActive(false)
                        audioSessionActive = false
                    } catch {
                        // placeholder for error handling
                    }
                    audioInterrupted = true
                }
            } else if interuptionVal == AVAudioSession.InterruptionType.ended {
                // [self endInterruption];
                if audioInterrupted {
                    let audioSession = AVAudioSession.sharedInstance()
                    do {
                        try audioSession.setActive(true)
                        audioSessionActive = true
                        if auAudioUnit.renderResourcesAllocated == false {
                            try auAudioUnit.allocateRenderResources()
                        }
                        try auAudioUnit.startHardware()
                        isRecording = true
                    } catch {
                        // placeholder for error handling
                    }
                }
            }
        }
    }
} // end of RecordAudio class
