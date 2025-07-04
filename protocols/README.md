We provide protocol generation scripts in this folder, each script is named after the database it processes.

Below is the supplementary information to our training set. This table is intended to help you verify whether your target training or testing data:

* is included in the training set of the AntiDeepfake models, or
* shares the same source data as the AntiDeepfake training set.

| Database            | Source Data                                                        | AntiDeepfake Training Set  Usage                       |
|---------------------|---------------------------------------------------------------------|---------------------------------------------------|
| AIShell3            | -                                                                   | Training: Train & Test sets<br>Validation: None   |
| ASVspoof2019-LA     | VCTK                                                                | Training: Train & Eval sets<br>Validation: Dev    |
| ASVspoof2021-LA     | VCTK                                                                | Training: All<br>Validation: None                 |
| ASVspoof2021-DF     | VCTK                                                                | Training: All<br>Validation: None                 |
| ASVspoof5           | MLS                                                                 | Training: Train & Test sets<br>Validation: Dev    |
| CFAD                | AIShell1&3, THCHS-30, MAGICDATA, Self-Recording                     | Training: Train & Test sets<br>Validation: Dev    |
| CNCeleb2            | -                                                                   | Training: All<br>Validation: None                 |
| Codecfake           | LibriTTS, VCTK, AIShell3                                            | Training: Train & Test sets<br>Validation: Dev    |
| CodecFake           | VCTK                                                                | Training: All<br>Validation: None                 |
| CVoiceFake          | Common Voice                                                        | Training: All<br>Validation: None                 |
| DECRO               | VCTK, AIShell1&2&3, Aidatatang_200zh, freeST, MagicData             | Training: Train & Test sets<br>Validation: Dev    |
| DFADD               | VCTK                                                                | Training: Train & Test sets<br>Validation: Valid  |
| Diffuse or Confuse  | LJSpeech                                                            | Training: All<br>Validation: None                 |
| DiffSSD             | LibriSpeech, LJSpeech                                               | Training: All<br>Validation: None                 |
| DSD                 | LibriSpeech, VCTK, AIHUB, Crowdsourced                              | Training: Train & Test sets<br>Validation: Dev    |
| FLEURS              | FLORES-101                                                          | Training: Train & Test sets<br>Validation: Dev    |
| FLEURS-R            | FLEURS                                                              | Training: Train & Test sets<br>Validation: Dev    |
| HABLA               | Crowdsourced                                                        | Training: All<br>Validation: None                 |
| LibriTTS            | -                                                                   | Training: Train & Test sets<br>Validation: Dev    |
| LibriTTS-R          | LibriTTS                                                             | Training: Train & Test sets<br>Validation: Dev    |
| LibriTTS-Vocoded    | LibriTTS                                                             | Training: Train & Test sets<br>Validation: Dev    |
| LJSpeech            | -                                                                   | Training: All<br>Validation: None                 |
| MLAAD               | M-AILABS                                                             | Training: All<br>Validation: None                 |
| MLS                 | -                                                                   | Training: Train & Test sets<br>Validation: Dev    |
| SpoofCeleb          | VoxCeleb1                                                 | Training: Train & Evaluation sets<br>Validation: Development |
| VoiceMOS            | Blizzard Challenge                                                  | Training: All<br>Validation: None                 |
| VoxCeleb2           | -                                                             | Training: All<br>Validation: None                 |
| VoxCeleb2-Vocoded   | VoxCeleb2                                                            | Training: All<br>Validation: None                 |
| WaveFake            | LJSpeech, JSUT                                                      | Training: All<br>Validation: None                 |
