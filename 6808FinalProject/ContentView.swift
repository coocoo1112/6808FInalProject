//
//  ContentView.swift
//  6808FinalProject
//
//  Created by Cooper Jones on 5/1/21.
//

import AVFoundation
import SwiftUI

var player: AVAudioPlayer?

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
        forResource: "chirp",
        withExtension: "mp3"
    ) else {
        return
    }

    recorder.startPlayback(soundFileURL: soundFileURL)
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
