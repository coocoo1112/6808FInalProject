//
//  ContentView.swift
//  6808FinalProject
//
//  Created by Cooper Jones on 5/1/21.
//

import AVFoundation
import SwiftUI
import SwiftUICharts
import MediaPlayer


var player: AVAudioPlayer?

struct ContentView: View {
    @State private var showDetails = false
    @ObservedObject var recorder: RecordAudio

    @State private var timer: DispatchSourceTimer!

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
            Text("Start playback and recording")
        }
        Button {
            playSound(recorder: recorder)

            let timeLimit = 40.0

            let queue = DispatchQueue(label: "MIT-6.808.-808FinalProject1.timer", qos: .userInteractive)
            timer = DispatchSource.makeTimerSource(flags: .strict, queue: queue)
            timer.schedule(deadline: .now() + timeLimit, leeway: .nanoseconds(0))
            timer.setEventHandler{
                stopRecording(recorder: recorder)
                print("stopped")
            }

            timer.activate()
        } label: {
            Text("Start 5s playback and recording")
        }
        Button {
            stopRecording(recorder: recorder)
        } label: {
            Text("Stop recording")
        }
        Button {
            // set volume to same thing every time
            let volumeView = MPVolumeView()
            let slider = volumeView.subviews.first(where: { $0 is UISlider }) as? UISlider

            DispatchQueue.main.asyncAfter(deadline: DispatchTime.now() + 0.01) {
                slider?.value = 0.7
            }
        } label: {
            Text("Set volume")
        }

        // expects a double array so we have to do this annoying conversion
//        LineView(data: recorder.fftPlot.map {Double($0)}, title: "FFT")
    }
}

func setupRecorder(recorder: RecordAudio) {
    recorder.setupAudioSessionForRecording()
}

func recordSound(recorder: RecordAudio) {
    recorder.startRecording()
}

func playSound(recorder: RecordAudio) {
    guard let soundFileURL = Bundle.main.url(
        forResource: "fmcw_chirp",
        withExtension: "wav"
    ) else {
        return
    }

    recorder.startPlayback(soundFileURL: soundFileURL)
}

func stopRecording(recorder: RecordAudio) {
    recorder.stopRecording()
}

// struct ContentView_Previews: PreviewProvider {
//    static var previews: some View {
//        ContentView(recorder: recorder)
//    }
// }
