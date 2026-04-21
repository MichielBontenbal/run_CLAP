```mermaid
graph TD
    %% Define Styles
    classDef main fill:#dae8fc,stroke:#6c8ebf,stroke-width:2px,font-weight:bold;
    classDef tower fill:#fff2cc,stroke:#d6b656,stroke-width:2px;
    classDef projection fill:#d5e8d4,stroke:#82b366,stroke-width:2px;
    classDef layer fill:#f8cecc,stroke:#b85450,stroke-width:1px;
    classDef output fill:#e1d5e7,stroke:#9673a6,stroke-width:2px,font-weight:bold;

    %% Top Level Model
    ClapModel("ClapModel"):::main;

    %% --- Text Branch ---
    subgraph text_model 
        direction TB
        T_Embed("embeddings (ClapTextEmbeddings)<br>- word_embeddings<br>- position_embeddings<br>- token_type_embeddings<br>- LayerNorm & Dropout"):::layer
        T_Encoder("encoder (ClapTextEncoder)<br>Contains 12x ClapTextLayer"):::layer
        T_Pooler("pooler (ClapTextPooler)<br>- dense (768 -> 768)<br>- Tanh"):::layer
        T_Embed --> T_Encoder --> T_Pooler;
    end

    %% --- Audio Branch ---
    subgraph audio_model 
        direction TB
        
        subgraph audio_encoder (ClapAudioEncoder)
            direction TB
            A_PatchEmbed("patch_embed (ClapAudioPatchEmbed)<br>proj: Conv2d (Spectrogram -> Patches)"):::layer
            
            A_Stage0("Stage 0: 2x ClapAudioLayer<br>dim: 128"):::layer
            A_DS0("downsample (PatchMerging)<br>dim: 128 -> 256"):::layer
            
            A_Stage1("Stage 1: 2x ClapAudioLayer<br>dim: 256"):::layer
            A_DS1("downsample (PatchMerging)<br>dim: 256 -> 512"):::layer

            A_Stage2("Stage 2: 12x ClapAudioLayer<br>dim: 512"):::layer
            A_DS2("downsample (PatchMerging)<br>dim: 512 -> 1024"):::layer

            A_Stage3("Stage 3: 2x ClapAudioLayer<br>dim: 1024"):::layer
            
            A_PatchEmbed --> A_Stage0 --> A_DS0 --> A_Stage1 --> A_DS1 --> A_Stage2 --> A_DS2 --> A_Stage3
        end
        
        A_Norm("norm (LayerNorm)"):::layer
        A_Pool("avgpool (AdaptiveAvgPool1d)"):::layer

        audio_encoder --> A_Norm --> A_Pool;
    end

    %% Projection Layers
    T_Proj("text_projection (ClapProjectionLayer)<br>- linear1 (768 -> 512)<br>- ReLU<br>- linear2 (512 -> 512)"):::projection
    A_Proj("audio_projection (ClapProjectionLayer)<br>- linear1 (1024 -> 512)<br>- ReLU<br>- linear2 (512 -> 512)"):::projection
    
    %% Final Outputs
    Text_Output[("Text Embedding<br>512-dim")]:::output
    Audio_Output[("Audio Embedding<br>512-dim")]:::output

    %% --- Connections ---
    ClapModel --> text_model:::tower;
    ClapModel --> audio_model:::tower;

    text_model -- "Text Features (768-dim)" --> T_Proj;
    audio_model -- "Audio Features (1024-dim)" --> A_Proj;
    
    T_Proj --> Text_Output;
    A_Proj --> Audio_Output;