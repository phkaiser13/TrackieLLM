# TrackieLLM: Documentação do Projeto

## 1. Conceito Central

**TrackieLLM** é uma plataforma de assistência multimodal projetada para operar em tempo real como um "companheiro de IA proativo". O núcleo do sistema é uma Inteligência Artificial que atua como um "cérebro", processando informações visuais e auditivas para analisar o contexto, compreender o ambiente e fornecer suporte inteligente ao usuário.

O objetivo principal é ampliar a percepção, promover a autonomia e garantir a segurança de pessoas com deficiência visual, revolucionando a forma como interagem com o mundo ao seu redor.

## 2. Pilares Tecnológicos

O TrackieLLM integra três áreas tecnológicas para criar uma percepção unificada do ambiente:

1.  **Visão Computacional:** Utiliza câmeras para "ver" e interpretar o mundo, reconhecendo rostos, objetos, textos, obstáculos e perigos.
2.  **Processamento de Áudio:** Captura e compreende comandos de voz (STT), sons do ambiente (VAD) e fornece feedback por áudio (TTS).
3.  **Inteligência Artificial (LLM):** Um modelo de linguagem grande (Large Language Model) atua como a unidade central de processamento, unindo as informações visuais e sonoras para fornecer um feedback contextualizado e útil em tempo real.

## 3. Benefícios e Diferenciais

*   **Autonomia Ampliada:** Permite que o usuário realize tarefas cotidianas com mais independência e explore novos ambientes com confiança.
*   **Segurança Proativa:** Detecta ativamente riscos como obstáculos, degraus, buracos e fumaça.
*   **Interação Natural:** A comunicação é feita por comandos de voz, tornando a experiência de uso fluida e intuitiva.
*   **Percepção Multimodal:** Integra dados de áudio, vídeo e sensores para uma compreensão completa do ambiente.
*   **Acessibilidade:** Posiciona-se como uma alternativa poderosa e de baixo custo a soluções comerciais caras.

## 4. Stack de Modelos de IA

O TrackieLLM é construído sobre um conjunto de modelos de IA otimizados para execução offline e em hardware com recursos limitados.

*   **IA Central (LLM):**
    *   **Modelo:** `Mistral-7B`
    *   **Formato:** GGUF (otimizado para `llama.cpp`)

*   **Visão Computacional:**
    *   **Detecção de Objetos:** `YOLOv5nu` (formato ONNX)
    *   **Análise de Profundidade e Navegação:** `DPT-SwinV2-Tiny-256` (MiDaS 3.1, ONNX, INT8) para detecção de passos, rampas, espaços livres e pontos de agarre.
    *   **Reconhecimento de Texto (OCR):** `Tesseract OCR` (via API nativa C++)

*   **Processamento de Áudio:**
    *   **Reconhecimento de Fala (ASR):** `whisper.cpp tiny.en` (formato GGML)
    *   **Ativação por Voz (Wake Word / VAD):** `Porcupine` e `Silero VAD`
    *   **Síntese de Voz (TTS):** `Piper` (Rhasspy) com vozes pré-treinadas em PT-BR.

## 5. Plataformas de Execução

### Ambientes de Produção (Uso Real)

O TrackieLLM foi projetado para rodar de forma nativa e otimizada nos seguintes sistemas:

*   **Hardware Embarcado:**
    *   **Orange Pi (8GB RAM + CUDA):** Plataforma principal.
    *   **Orange Pi (RISC-V, 8GB RAM + CUDA):** Plataforma secundária de alta praticidade.
    *   **Raspberry Pi / Orange Pi (Modelos de 8-32GB RAM):** Para desenvolvimento e uso pela comunidade.
*   **Dispositivos Móveis (via app `TrackWay`):**
    *   **Android:** Suporte nativo.
    *   **iOS:** Suporte nativo com alta prioridade, otimizado para o acelerador gráfico **Metal**.
*   **Desktop (via terminal `TrackWay`):**
    *   **Linux:** Suporte a CUDA e ROCm.

### Ambientes de Teste e Apresentação (via `Trackie Studio`)

*   **Windows, macOS e Linux:** Para fins de demonstração, testes e treinamento de modelos.

## 6. Compilação e Implantação

*   Nos aplicativos **Trackie Studio** (Desktop) e **TrackWay** (Mobile), o núcleo do TrackieLLM deve ser compilado como uma biblioteca dinâmica (`.dll`, `.so`, `.dylib`, etc.).
*   Nos dispositivos embarcados (Orange/Raspberry Pi), o sistema pode rodar como um executável nativo direto no sistema operacional (com kernel modificado) ou dentro de um contêiner para portabilidade.

## 7. Missão do Projeto

> Levar acessibilidade inteligente a ambientes educacionais, industriais e ao dia a dia por meio de IA de ponta e hardware acessível.
