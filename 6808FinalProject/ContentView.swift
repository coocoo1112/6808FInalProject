//
//  ContentView.swift
//  6808FinalProject
//
//  Created by Cooper Jones on 5/1/21.
//

import AVFoundation
import SwiftUI

var player: AVAudioPlayer? = nil

struct ContentView: View {
    @State private var showDetails = false
    var body: some View {
        Text("Hello, world!")
            .padding()
        Button {
//            print("Image tapped!")
            playSound()
        } label: {
            Text("Play Sound")
        }
    }
    
    
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
            AVAudioSession.Category.playback
        )

        try AVAudioSession.sharedInstance().setActive(true)

        // Play a sound
        player = try AVAudioPlayer(
            contentsOf: soundFileURL
        )

        player?.play()
    }
    catch {
        // Handle error
        print("oh shit")
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
