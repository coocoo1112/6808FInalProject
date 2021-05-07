//
//  ContentView.swift
//  6808FinalProject
//
//  Created by Cooper Jones on 5/1/21.
//

import AVFoundation
import SwiftUI
import SwiftUICharts


var player: AVAudioPlayer?
var timer: DispatchSourceTimer!

struct ContentView: View {
    @State private var showDetails = false
    @ObservedObject var recorder: RecordAudio

    var body: some View {
        Text("Hello, world!")
            .padding()
        Button {
            recordSound(recorder: recorder)
        } label: {
            Text("Start recording")
        }
        Button {
            
            playSound(recorder: recorder)
        } label: {
            Text("Start playback")
        }

        // expects a double array so we have to do this annoying conversion
        LineView(data: recorder.fftPlot.map {Double($0)}, title: "FFT")
    }
}

func setupRecorder(recorder: RecordAudio) {
    recorder.setupAudioSessionForRecording()
}

func recordSound(recorder: RecordAudio) {
    recorder.startRecording(startTime: CFAbsoluteTimeGetCurrent())
}

func PCMBufferToFloatArray(_ pcmBuffer: AVAudioPCMBuffer) -> [[Float]]? {
    if let floatChannelData = pcmBuffer.floatChannelData {
        let channelCount = Int(pcmBuffer.format.channelCount)
        let frameLength = Int(pcmBuffer.frameLength)
        let stride = pcmBuffer.stride
        var result: [[Float]] = Array(repeating: Array(repeating: 0.0, count: frameLength), count: channelCount)
        for channel in 0..<channelCount {
            for sampleIndex in 0..<frameLength {
                result[channel][sampleIndex] = floatChannelData[channel][sampleIndex * stride]
            }
        }
        return result
    } else {
        print("format not in Float")
        return nil
    }
}

func segment(of buffer: AVAudioPCMBuffer, from startFrame: AVAudioFramePosition, to endFrame: AVAudioFramePosition) -> AVAudioPCMBuffer? {
    let framesToCopy = AVAudioFrameCount(endFrame - startFrame)
    guard let segment = AVAudioPCMBuffer(pcmFormat: buffer.format, frameCapacity: framesToCopy) else { return nil }

    let sampleSize = buffer.format.streamDescription.pointee.mBytesPerFrame

    let srcPtr = UnsafeMutableAudioBufferListPointer(buffer.mutableAudioBufferList)
    let dstPtr = UnsafeMutableAudioBufferListPointer(segment.mutableAudioBufferList)
    for (src, dst) in zip(srcPtr, dstPtr) {
        memcpy(dst.mData, src.mData?.advanced(by: Int(startFrame) * Int(sampleSize)), Int(framesToCopy) * Int(sampleSize))
    }

    segment.frameLength = framesToCopy
    return segment
}

func playSound(recorder: RecordAudio) {
    print("test")
    var start: AVAudioFramePosition!
    var end: AVAudioFramePosition!
    var timer: DispatchSourceTimer!
    guard let soundFileURL = Bundle.main.url(
        forResource: "chirp3",
        withExtension: "wav"
    ) else {
        return
    }
    start=0
    end=221
    let file = try! AVAudioFile(forReading: soundFileURL)
    let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: file.fileFormat.sampleRate, channels: 1, interleaved: false)
    let buf = AVAudioPCMBuffer(pcmFormat: format!, frameCapacity: 1024)!
    try! file.read(into: buf)
    let shortBuffer = segment(of: buf, from: start, to: end)
    let testArray = PCMBufferToFloatArray(buf)
    let shortArray = PCMBufferToFloatArray(shortBuffer!)
    let shortFloatArray = UnsafeBufferPointer(start: buf.floatChannelData![0], count:Int(shortBuffer!.frameLength))
    let floatArray = UnsafeBufferPointer(start: buf.floatChannelData![0], count:Int(buf.frameLength))
    print("floatArray \(shortFloatArray)\n")
//    for point in shortFloatArray {
//        print("\(point)")
//    }
    //recorder.startPlayback(soundFileURL: soundFileURL)
    startTimer(recorder: recorder, soundFileURL: soundFileURL, buf: shortBuffer!)
    DispatchQueue.main.asyncAfter(deadline: .now() + 10) {
       stopTimer()
        print("stopped")
    }
//    let queue = DispatchQueue(label: "com.domain.app.timer", qos: .userInteractive)
//    timer = DispatchSource.makeTimerSource(flags: .strict, queue: queue)
//    timer.schedule(deadline: .now(), repeating: 1.0001, leeway: .nanoseconds(0))
//    timer.setEventHandler {
//        print("test")
//    }
//    for i in 1...10{
//        recorder.startPlayback(soundFileURL: soundFileURL)
//    }
    
}

private func timestep(recorder: RecordAudio, soundFileURL: URL, buf: AVAudioPCMBuffer){
    recorder.pauseRecording()
    print("paused")
    //recorder.startPlayback(soundFileURL: soundFileURL)
    recorder.startPlayback(soundFileURL: soundFileURL)
    let startTime = CFAbsoluteTimeGetCurrent()
    let second: Double = 1000000
    usleep(useconds_t(0.5 * second))
    print(CFAbsoluteTimeGetCurrent() - startTime)
    recorder.resumeRecording()
    print(CFAbsoluteTimeGetCurrent() - startTime)
//    DispatchQueue.main.asyncAfter(deadline: .now() + 0.003) {
//        print(CFAbsoluteTimeGetCurrent() - startTime)
//        recorder.resumeRecording()
//        print(CFAbsoluteTimeGetCurrent() - startTime)
//    }
}

private func startTimer(recorder: RecordAudio, soundFileURL: URL, buf: AVAudioPCMBuffer){
    var i = 1
    let queue = DispatchQueue(label: "com.firm.app.timer", attributes: .concurrent)

    timer = DispatchSource.makeTimerSource(queue: queue)

    timer.setEventHandler { //[weak self] in // `[weak self]` only needed if you reference `self` in this closure and you want to prevent strong reference cycle
        print("test" + " " + String(i))
        i += 1
        timestep(recorder: recorder, soundFileURL: soundFileURL, buf: buf)
    }

    timer.schedule(deadline: .now(), repeating: .milliseconds(1000), leeway: .nanoseconds(0))

    timer.resume()
}

private func stopTimer() {
    timer = nil
}

func playSound() {
    guard let soundFileURL = Bundle.main.url(
        forResource: "chirp",
        withExtension: "mp3"
    ) else {
        return
    }

    print(Bundle.main.bundleURL)

    do {
        // Configure and activate the AVAudioSession
        try AVAudioSession.sharedInstance().setCategory(
            AVAudioSession.Category.playAndRecord
        )

        try AVAudioSession.sharedInstance().setActive(true)

        // Play a sound
        player = try AVAudioPlayer(
            contentsOf: soundFileURL
        )
        player?.play()
        
    } catch {
        // Handle error
        print("oh shit")
    }
}

// struct ContentView_Previews: PreviewProvider {
//    static var previews: some View {
//        ContentView(recorder: recorder)
//    }
// }
