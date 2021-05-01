//
//  ContentView.swift
//  6808FinalProject
//
//  Created by Cooper Jones on 5/1/21.
//

import SwiftUI

struct ContentView: View {
    @State private var showDetails = false
    var body: some View {
        Text("Hello, world!")
            .padding()
        Button {
            print("Image tapped!")
        } label: {
            Text("Play Sound")
        }
    }
    
    
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
