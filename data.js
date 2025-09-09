const datasets = [
  {
    name: "ATCOSIM",
    prompt: "Generate an utterance for air traffic control: A pilot requests descent clearance at Munich airport.",
    generated_text: "Captain James requests descent clearance from Munich tower.",
    tts_audio: "audio/atcosim_original.wav",
    respelling: "Cap-tin Jeymz re-kwests desent kleer-ans from Myoo-nik tower.",
    respelling_audio: "audio/atcosim_respelling.wav"
  },
  {
    name: "ATCO2",
    prompt: "Create an aircraft communication involving weather update.",
    generated_text: "Tower reports wind speed and direction for incoming flight.",
    tts_audio: "audio/atco2_original.wav",
    respelling: "Tow-er reports wind speed and direction for incoming flight.",
    respelling_audio: "audio/atco2_respelling.wav"
  },
  {
    name: "Court",
    prompt: "Generate a court dialogue: The judge instructs the witness.",
    generated_text: "The judge instructs the witness to answer truthfully.",
    tts_audio: "audio/court_original.wav",
    respelling: "The juhj instructs the wit-ness to ans-er trooth-fully.",
    respelling_audio: "audio/court_respelling.wav"
  },
  {
    name: "MedSyn",
    prompt: "Create a medical prescription sentence.",
    generated_text: "Take two tablets of paracetamol every six hours after meals.",
    tts_audio: "audio/medsyn_original.wav",
    respelling: "Take too tab-lits of para-seet-a-mol every siks hours after meals.",
    respelling_audio: "audio/medsyn_respelling.wav"
  }
];
