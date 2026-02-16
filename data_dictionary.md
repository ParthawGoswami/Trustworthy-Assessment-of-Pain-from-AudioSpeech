# Data Dictionary

This Data Dictionary provides detailed information about each variable in the TAME Pain Dataset. It includes descriptions, units, and coding schemes. For data collection protocol and additional details the user should consult with the TAME Pain data release publication. 

## Table of Contents

1. [Audio Recordings](#audio-recordings)
2. [Metadata Files](#metadata-files)
   - [Audio Metadata (`meta_audio.csv`)](#audio-metadata-meta_audiocsv)
   - [Participant Data (`meta_participant.csv`)](#participant-data-meta_participantcsv)
3. [Annotations](#annotations)
   - [External Disturbances (`External_Disturbances.csv`)](#external-disturbances-external_disturbancescsv)
   - [Speech Errors and Disturbances (`Speech_Errors_and_Disturbances.csv`)](#speech-errors-and-disturbances-speech_errors_and_disturbancescsv)
   - [Audio Cut Out (`Audio_Cut_Out.csv`)](#audio-cut-out-audio_cut_outcsv)
   - [Audible Breath (`Audible_Breath.csv`)](#audible-breath-audible_breathcsv)
   - [No Pain Rating So Copied (`No_Pain_Rating_So_Copied.csv`)](#no-pain-rating-so-copied-no_pain_rating_so_copiedcsv)
   - [No Assigned Sentence (`No_Assigned_Sentence.csv`)](#no-assigned-sentence-no_assigned_sentencecsv)
   - [No Pain Rating (`No_Pain_Rating.csv`)](#no-pain-rating-no_pain_ratingcsv)

---

## Audio Recordings

- **File Naming Convention:** `PID.COND.UTTNUM.UTTID.wav`
  - **PID:** Participant Identification (e.g., `p12345`)
  - **COND:** Experimental Condition (`LC`, `LW`, `RC`, `RW`)
  - **UTTNUM:** Utterance Number (sequential per condition)
  - **UTTID:** Utterance ID (corresponds to assigned sentence or `99999` for pain statements)

## Metadata Files

### Audio Metadata (`meta_audio.csv`)

| Variable Name  | Description                                                                                   | Units             | Codes/Values                                                                 |
|----------------|-----------------------------------------------------------------------------------------------|-------------------|------------------------------------------------------------------------------|
| PID            | Participant Identification                                                                    | N/A               | `pXXXXX` (e.g., `p12345`)                                                   |
| COND           | Experimental Condition                                                                       | N/A               | `LC` (Left Cold), `LW` (Left Warm), `RC` (Right Cold), `RW` (Right Warm)    |
| UTTNUM         | Utterance Number within each condition                                                        | Integer           | Sequential numbering starting at 1 per condition                             |
| UTTID          | Utterance ID corresponding to assigned sentence or pain statement                             | Integer          | Assigned sentence ID from Appendix or `99999` for pain statements           |
| PAIN LEVEL     | Raw self-reported pain level extracted from audio data                                       | Integer           | `0` to `10` (original scale)                                                 |
| REVISED PAIN   | Modified pain level aligned with the 1-10 scale                                               | Integer           | `1` to `10` (with original `0` relabeled to `1`)                            |
| DURATION       | Duration of the audio file                                                                    | Seconds           | Numeric value (e.g., `2.65`)                                                 |
| ACTION LABEL   | Quality rating of the audio based on disturbances and errors                                 | Integer           | `0` (Highest quality) to `4` (Lowest quality)                               |
| NOTES          | Manual annotations and comments regarding audio disturbances, errors, etc.                   | Text              | Descriptive notes, multiple annotations separated by semicolons (`;`)       |

### Participant Data (`meta_participant.csv`)

| Variable Name    | Description                                                                                                 | Units        | Codes/Values                                                                                   |
|------------------|-------------------------------------------------------------------------------------------------------------|--------------|------------------------------------------------------------------------------------------------|
| PID              | Participant Identification                                                                                  | N/A          | `pXXXXX` (e.g., `p12345`)                                                                       |
| GENDER           | Self-reported gender                                                                                        | Categorical  | `Man`, `Woman`, `Non-Binary`, `Prefer to self-describe`                                        |
| AGE              | Self-reported age from the screening survey                                                                 | Years        | Integer values (e.g., `21`)                                                                      |
| RACE/ETHNICITY   | Self-reported race/ethnicity from the screening survey                                                      | Categorical  | `Hispanic/Latino`, `American Indian or Alaska Native`, `Asian`, `Black or African American`, `Native Hawaiian or Other Pacific Islander`, `White`, `Two or More Races` |
| FOLDER SIZE      | Digital storage size of all audio files for the participant                                                 | Megabytes    | Numeric value (e.g., `11.72`)                                                                    |
| NUMBER OF FILES  | Total count of audio files for the participant                                                               | Integer      | Numeric value (e.g., `138`)                                                                       |
| TOTAL DURATION   | Sum of durations of all audio files for the participant                                                     | Seconds      | Numeric value (e.g., `365.90`)                                                                    |
| LC               | Completion status of the "Left Cold" condition                                                               | Binary       | `1` = Completed, `0` = Incomplete                                                                |
| LW               | Completion status of the "Left Warm" condition                                                               | Binary       | `1` = Completed, `0` = Incomplete                                                                |
| RC               | Completion status of the "Right Cold" condition                                                              | Binary       | `1` = Completed, `0` = Incomplete                                                                |
| RW               | Completion status of the "Right Warm" condition                                                              | Binary       | `1` = Completed, `0` = Incomplete                                                                |

---

## Annotations

Each annotation file contains the following common variables:

| Variable Name  | Description                                                                 | Units        | Codes/Values                                                   |
|----------------|-----------------------------------------------------------------------------|--------------|----------------------------------------------------------------|
| PID            | Participant Identification                                                  | N/A          | `pXXXXX` (e.g., `p12345`)                                     |
| COND           | Experimental Condition                                                     | N/A          | `LC`, `LW`, `RC`, `RW`                                         |
| UTTNUM         | Utterance Number within each condition                                      | Integer      | Sequential numbering starting at 1 per condition               |
| UTTID          | Utterance ID corresponding to assigned sentence or pain statement           | Integer   | Assigned sentence ID or `99999` for pain statements            |
| PAIN LEVEL     | Raw self-reported pain level                                                | Integer      | `0` to `10`                                                    |
| REVISED PAIN   | Modified pain level aligned with the 1-10 scale                             | Integer      | `1` to `10`                                                    |
| NOTES          | Specific annotation details                                                 | Text         | Descriptive notes related to the annotation category           |
| ACTION LABEL   | (Except for `No_Pain_Rating.csv`) Quality rating of the audio              | Integer      | `0` to `4`                                                      |
| NOISE RELATION | (Only in `External_Disturbances.csv`) Source of external disturbance      | Categorical  | `foreground`, `background`, `foreground and background`        |

### External Disturbances (`External_Disturbances.csv`)

| Variable Name  | Description                                                                                       | Units | Codes/Values                                                   |
|----------------|---------------------------------------------------------------------------------------------------|-------|----------------------------------------------------------------|
| PID            | Participant Identification                                                                        | N/A   | `pXXXXX` (e.g., `p12345`)                                     |
| COND           | Experimental Condition                                                                           | N/A   | `LC`, `LW`, `RC`, `RW`                                         |
| UTTNUM         | Utterance Number within each condition                                                            | Integer | Sequential numbering starting at 1 per condition               |
| UTTID          | Utterance ID corresponding to assigned sentence or pain statement                                 | Integer    | Assigned sentence ID or `99999` for pain statements            |
| PAIN LEVEL     | Raw self-reported pain level                                                                      | Integer | `0` to `10`                                                    |
| REVISED PAIN   | Modified pain level aligned with the 1-10 scale                                                   | Integer | `1` to `10`                                                    |
| NOTES          | Details about the external disturbance (intensity, noise type, location)                           | Text    | e.g., "loud beep at end"; multiple annotations separated by `;`|
| ACTION LABEL   | Quality rating of the audio                                                                      | Integer | `0` to `4`                                                      |
| NOISE RELATION | Source of the external disturbance                                                               | Categorical | `foreground`, `background`, `foreground and background`        |

### Speech Errors and Disturbances (`Speech_Errors_and_Disturbances.csv`)

| Variable Name  | Description                                                              | Units | Codes/Values                                  |
|----------------|--------------------------------------------------------------------------|-------|-----------------------------------------------|
| PID            | Participant Identification                                               | N/A   | `pXXXXX` (e.g., `p12345`)                    |
| COND           | Experimental Condition                                                  | N/A   | `LC`, `LW`, `RC`, `RW`                        |
| UTTNUM         | Utterance Number within each condition                                   | Integer | Sequential numbering starting at 1 per condition|
| UTTID          | Utterance ID corresponding to assigned sentence or pain statement        | Integer    | Assigned sentence ID or `99999` for pain statements |
| PAIN LEVEL     | Raw self-reported pain level                                             | Integer | `0` to `10`                                   |
| REVISED PAIN   | Modified pain level aligned with the 1-10 scale                          | Integer | `1` to `10`                                   |
| NOTES          | Details about speech errors or disturbances (type, affected words, location)| Text    | e.g., "stutter on 'pain'"          |
| ACTION LABEL   | Quality rating of the audio                                             | Integer | `0` to `4`                                     |

### Audio Cut Out (`Audio_Cut_Out.csv`)

| Variable Name  | Description                                                          | Units | Codes/Values                        |
|----------------|----------------------------------------------------------------------|-------|-------------------------------------|
| PID            | Participant Identification                                           | N/A   | `pXXXXX` (e.g., `p12345`)          |
| COND           | Experimental Condition                                              | N/A   | `LC`, `LW`, `RC`, `RW`              |
| UTTNUM         | Utterance Number within each condition                               | Integer | Sequential numbering starting at 1 per condition|
| UTTID          | Utterance ID corresponding to assigned sentence or pain statement    | Integer    | Assigned sentence ID or `99999` for pain statements |
| PAIN LEVEL     | Raw self-reported pain level                                         | Integer | `0` to `10`                         |
| REVISED PAIN   | Modified pain level aligned with the 1-10 scale                      | Integer | `1` to `10`                         |
| NOTES          | Details about the audio cut (location, missing words)                | Text    | e.g., "first word 'The' cut out"|
| ACTION LABEL   | Quality rating of the audio                                         | Integer | `0` to `4`                           |

### Audible Breath (`Audible_Breath.csv`)

| Variable Name  | Description                                               | Units | Codes/Values                        |
|----------------|-----------------------------------------------------------|-------|-------------------------------------|
| PID            | Participant Identification                                | N/A   | `pXXXXX` (e.g., `p12345`)          |
| COND           | Experimental Condition                                   | N/A   | `LC`, `LW`, `RC`, `RW`              |
| UTTNUM         | Utterance Number within each condition                    | Integer | Sequential numbering starting at 1 per condition|
| UTTID          | Utterance ID corresponding to assigned sentence or pain statement | Integer   | Assigned sentence ID or `99999` for pain statements |
| PAIN LEVEL     | Raw self-reported pain level                              | Integer | `0` to `10`                         |
| REVISED PAIN   | Modified pain level aligned with the 1-10 scale           | Integer | `1` to `10`                         |
| NOTES          | Details about audible breaths (type, location)            | Text    | e.g., "audible inhale at middle"    |
| ACTION LABEL   | Quality rating of the audio                                | Integer | `0` to `4`                           |

### No Pain Rating So Copied (`No_Pain_Rating_So_Copied.csv`)

| Variable Name  | Description                                                    | Units | Codes/Values                              |
|----------------|----------------------------------------------------------------|-------|-------------------------------------------|
| PID            | Participant Identification                                     | N/A   | `pXXXXX` (e.g., `p12345`)                |
| COND           | Experimental Condition                                        | N/A   | `LC`, `LW`, `RC`, `RW`                    |
| UTTNUM         | Utterance Number within each condition                         | Integer | Sequential numbering starting at 1 per condition|
| UTTID          | Utterance ID corresponding to assigned sentence or pain statement  | Integer   | Assigned sentence ID or `99999` for pain statements |
| PAIN LEVEL     | Raw self-reported pain level                                    | Integer | `0` to `10`                               |
| REVISED PAIN   | Modified pain level aligned with the 1-10 scale                 | Integer | `1` to `10`                               |
| NOTES          | Details about pain level copying (source file ID)               | Text    | e.g., "no pain rating given so copied down rather than up"        |
| ACTION LABEL   | Quality rating of the audio                                    | Integer | `0` to `4`                                 |

### No Assigned Sentence (`No_Assigned_Sentence.csv`)

| Variable Name  | Description                                          | Units | Codes/Values                                 |
|----------------|------------------------------------------------------|-------|----------------------------------------------|
| PID            | Participant Identification                           | N/A   | `pXXXXX` (e.g., `p12345`)                   |
| COND           | Experimental Condition                              | N/A   | `LC`, `LW`, `RC`, `RW`                       |
| UTTNUM         | Utterance Number within each condition               | Integer | Sequential numbering starting at 1 per condition|
| UTTID          | Utterance ID corresponding to assigned sentence or pain statement | Integer   | Assigned sentence ID or `99999` for pain statements |
| PAIN LEVEL     | Raw self-reported pain level                          | Integer | `0` to `10`                                 |
| REVISED PAIN   | Modified pain level aligned with the 1-10 scale       | Integer | `1` to `10`                                 |
| NOTES          | Details about missing assigned sentence (extra dialogue)| Text    |                                           |
| ACTION LABEL   | Quality rating of the audio                            | Integer | `0` to `4`                                   |

### No Pain Rating (`No_Pain_Rating.csv`)

| Variable Name  | Description                                  | Units | Codes/Values                     |
|----------------|----------------------------------------------|-------|----------------------------------|
| PID            | Participant Identification                   | N/A   | `pXXXXX` (e.g., `p12345`)       |
| COND           | Experimental Condition                      | N/A   | `LC`, `LW`, `RC`, `RW`           |
| UTTNUM         | Utterance Number within each condition       | Integer | Sequential numbering starting at 1 per condition|
| UTTID          | Utterance ID corresponding to assigned sentence or pain statement | Integer   | Assigned sentence ID or `99999` for pain statements |
| PAIN LEVEL     | Raw self-reported pain level                  | Integer | `0` to `10`                      |
| REVISED PAIN   | Modified pain level aligned with the 1-10 scale | Integer | `1` to `10`                      |
| NOTES          | Indicates absence of pain rating and no source to copy from | Text    | e.g., "no pain rating available"|
| ACTION LABEL   | Not applicable (files excluded from pain rating) | N/A    | N/A                              |

---

## Variable Definitions

### Action Label (`ACTION LABEL`)

A discrete scale from `0` to `4` indicating the quality of the audio file:

- `0`: Clean audio free from disturbances or errors.
- `1`: Audio with minor issues, such as low-severity disturbances or single-word cuts.
- `2`: Audio with moderate issues, including low-intensity external disturbances.
- `3`: Audio with significant issues, such as moderate-severity disturbances.
- `4`: Lowest quality audio with high potential to confound data processing.

### Noise Relation (`NOISE RELATION`)

Indicates the general source of external disturbances:

- `foreground`: Likely related to the speaker's movements or environment.
- `background`: Unrelated to the speaker, e.g., external noise sources.
- `foreground and background`: Contains both foreground and background disturbances.

---

## Notes

- **Multiple Annotations:** If multiple annotations apply to a single audio file, the file will appear in multiple annotation CSVs. The `NOTES` column will contain all relevant annotations separated by semicolons (`;`).

- **Pain Level Adjustment:** Original pain levels of `0` have been relabeled to `1` to maintain consistency within the 1-10 pain scale.

- **Excluded Utterances:** Five utterances without any pain rating have been excluded from all analyses and are listed only in `No_Pain_Rating.csv`.

- **Background Fan Noise:** All audio recordings contain a uniform background fan noise, which varies in intensity across participants but is not annotated within the dataset.

- **Annotation Limitation:** Annotations are potentially subject to human error since they were manually reviewed by a single author, TD. Furthermore, TD was not blinded to the pain levels or conditions reported in each audio file during the annotation process. See the data release publication for more details. 

---