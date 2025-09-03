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

	Trackie:
    Possui uma i.a Central, como nucleo ou como "Cerebro", procesa informações inteligentes, analisa contextos, analiza imagens, **Funciona quase em tempo real**, como se fosse sua alexa pessoal.
	Possui modelos como: yolov5nu (para detecção de objetos, especialmente os perigosos), intel-midas ultra leve (a principal é medir distancias para **PASSSOS**. Free-space / Traversability + Step / Ramp detector (navegação e risco) e por fim Affordance / Grasp-point detector (onde agarrar)).
	Fora modelos de OCR, fala, detecção de fala, tts, stt.

		### Análise Completa da Aplicação Trackie

		#### O Conceito Central: Uma Janela Inteligente para o Mundo

		Trackie é apresentado como um "companheiro de IA proativo". A ideia fundamental não é apenas criar um aplicativo, mas sim uma plataforma completa de **assistência multimodal em tempo real**. O objetivo principal é ampliar a percepção do usuário, promover sua autonomia e, acima de tudo, garantir sua segurança no dia a dia.

		Para atingir esse objetivo, o Trackie foi projetado para "ver, ouvir e compreender o ambiente ao redor", oferecendo ao usuário um suporte inteligente e totalmente contextualizado com a sua situação.

		#### A Revolução na Interação com o Ambiente

		O projeto se posiciona como uma "revolução na forma como pessoas com deficiência visual interagem com o ambiente". A grande inovação está na combinação de três pilares tecnológicos:

		1.  **Visão Computacional:** Permite que o sistema "veja" e interprete o mundo através de uma câmera.
		2.  **Processamento de Áudio:** Captura e entende comandos de voz e sons do ambiente.
		3.  **Inteligência Artificial:** É o cérebro que une as informações visuais e sonoras para fornecer um feedback útil e em tempo real.

		Graças a essa combinação, o Trackie é capaz de **reconhecer rostos, objetos, textos, obstáculos e perigos iminentes**. Com base nesse reconhecimento, ele fornece um retorno (feedback) inteligente ao usuário, capacitando-o com um nível de autonomia e segurança muito maior.

		#### Principais Benefícios e Diferenciais (Por que usar o Trackie?)

		O documento destaca quatro razões principais para se utilizar a plataforma:

		* **Autonomia Ampliada:** O usuário ganha a capacidade de realizar tarefas cotidianas com mais independência e de explorar novos ambientes com maior confiança e facilidade.
		* **Segurança Proativa:** O sistema não é apenas reativo; ele ativamente procura por riscos. Ele pode detectar obstáculos no caminho, mudanças de nível (como degraus ou buracos), fumaça (indicando um possível incêndio) e outros perigos em potencial.
		* **Interação Natural:** A comunicação com o Trackie é feita por meio de **comandos de voz naturais**, tornando a experiência de uso mais intuitiva e fluida, sem a necessidade de interfaces complexas.
		* **Percepção Multimodal:** O sistema não depende de uma única fonte de informação. Ele **integra áudio, vídeo e dados de sensores** para criar uma compreensão muito mais completa e profunda do ambiente ao redor do usuário.

		Um dos maiores diferenciais mencionados é que o Trackie se posiciona como uma **alternativa acessível e poderosa a soluções comerciais de alto custo**, como o OrCam MyEye.


		#### Missão do Projeto

		A missão do Trackie é clara e ambiciosa: "levar acessibilidade inteligente a ambientes educacionais, industriais e ao dia a dia por meio de IA de ponta e hardware acessível".

	# Modelos escolhidos:
	yolov5nu (onnx);
	Mistral-7B (GGUF pronto para llama.cpp / TheBloke)
	DPT-SwinV2-Tiny-256 (MiDaS 3.1, versão tiny) — convertida para ONNX + INT8;
	Tesseract OCR (C++ API nativa, offline).
	
	ASR: whisper.cpp tiny.en (ggml);
	Wake / VAD: Porcupine (wakeword) + Silero VAD;
	TTS: Piper (rhasspy/piper) + voz PT-BR pré-treinada (ex.: pt_BR-faber-medium ou pt_BR-edresson-low);
	
# Prática real (Onde o produto roda de fato.)

Orange PI (8gb de ram + cuda) --> Modelo escolhido para ser o hospedeiro principal do trackieLLM, kernel e OS próprios baseados em linux.
Orange PI (risc-v) + (8gb de ram + cuda) --> Modelo escolhido para ser o hospedeiro mais prático do trackieLLM, kernel e OS próprios baseados em linux.
Raspberry pi ou Orange Pi (Modelos mais potentes com 8-32gb de ram) --> Pode ser testado e usado para desenvolver trackies pela comunidade.)

Android --> Roda via TrackWay direto no celular, pronto para prática.
IOS --> Roda via TrackWay direto no celular, pronto para prática, Suporte real e completamente desenvolvido para metal e outros aceleradores reais da apple, prioridade está aqui.
Linux --> Roda via terminal (TrackWay) (Suporte a cuda opcional, incentivado e bem desenvolvido para rocm, depende do sistema e da necessidade do usuario.)


# Não praticiais (apresentação e/ou teste, treinamento)

Windows --> Roda via Trackie Studio (Suporte a cuda opcional, depende do sistema e da necessidade do usuario.)
MacOS --> Roda via Trackie Studio (Suporte a metal opcional e de teste, depende do sistema e da necessidade do usuario.)
Linux --> Roda via Trackie Studio (Suporte a cuda opcional, incentivado e bem desenvolvido para rocm, depende do sistema e da necessidade do usuario.)

freebsd e outros não possuem suporte.




 **ESSE documentão detalhado de todos os folders do TrackieLLM**, ajuda explicando **cada pasta e subpasta**, incluindo as subpastas Rust ou equivalentes, seguindo o modelo de descrição.

---

### 📂 TrackieLLM - Estrutura de Folders

#### Top-level

* **README.md** – visão geral do projeto, instruções de build e quickstart.
* **LICENSE** – licença do projeto (definida posteriormente).
* **CONTRIBUTING.md** – guia de contribuição, padrões de commits e PRs.
* **CHANGELOG.md** – histórico de mudanças por versão.
* **CODE\_OF\_CONDUCT.md** – conduta para colaboradores.
* **SECURITY.md** – políticas de segurança e reporte de vulnerabilidades.
* **.gitignore** – arquivos/pastas ignorados pelo Git.
* **.clang-format / rustfmt.toml** – formatação padronizada para C/C++ e Rust.
* **.github/** – workflows, templates de issues e PRs.

---


#### 📂 src/ – código-fonte principal

* **monitoring/** – coleta de métricas, performance e health checks.

  * **C files** – implementações core.
  * **src/** (Rust) – telemetry, metrics collectors e integração com C.
* **security/** – autenticação, criptografia e canais seguros.

  * **C files** – core security modules.
  * **src/** (Rust) – key management e secure channel abstractions.
* **deployment/** – atualizações, pacotes e versionamento.

  * **C files** – updater, installer.
  * **src/** (Rust) – version checker e package manager.
* **experiments/** – benchmarking, testes de modelos, análise de métricas.

  * **C files** – runner de benchmarks.
  * **src/** (Rust) – model analysis, metrics comparators.
* **internal\_tools/** – parser de configs, gerenciador de arquivos, utils gerais.

  * **C files** – parsing e file management.
  * **src/** (Rust) – config loader, filesystem utils.
* **logging\_ext/** – logging de eventos e auditoria.

  * **C files** – loggers principais.
  * **src/** (Rust) – formatadores, helpers para auditoria.
* **memory/** – gerenciamento avançado de memória, tracking, garbage collection.

  * **C files** – pools, trackers.
  * **src/** (Rust) – allocators, garbage collection helpers.
* **ai\_models/** – loaders e runners de modelos ONNX/GGUF, integração com núcleo.

  * **C/C++ files** – runners e loaders.
  * **src/** (Rust) – integração com modelos, runners GGUF/ONNX.
* **networking/** – gerenciamento de sockets, protocolos e pools de conexão.

  * **C files** – network manager, socket handler.
  * **src/** (Rust) – protocol logic, connection pooling.
* **async\_tasks/** – scheduler de tasks, thread pool e executor async.

  * **C files** – task scheduler, worker pool.
  * **src/** (Rust) – task manager, async executor.
* **gpu/extensions/** – operações adicionais de tensor e imagem para CUDA, ROCm e Metal.

  * **cuda/** – C/CUDA kernels, tensor/image ops.
  * **rocm/** – C++ ROCm ops.
  * **metal/** – Metal ops para Apple.
* **integration/** – bridge para plugins externos, APIs e módulos embarcados.

  * **C/C++ files** – external interface.
  * **src/** (Rust) – bridge e plugin manager.
* **profiling/** – profiling de CPU, GPU e memória; coleta de métricas.

  * **C files** – profiler core e memory profiler.
  * **src/** (Rust) – profiler logic, metrics collector.
* **cortex/** – núcleo de raciocínio/contextual e engine de decisão.

  * **C files** – decision engine, contextual reasoner.
  * **rust/** – reasoning, memory manager.
* **vision/** – pipeline de visão, profundidade, detecção de objetos e OCR.

  * **C/C++ files** – pipelines e detector implementations.
  * **src/** (Rust) – depth processing, object analysis.
* **audio/** – pipeline de áudio, ASR (Whisper), TTS (Piper).

  * **C files** – core audio pipelines.
  * **src/** (Rust) – asr\_processing, tts\_synthesis.
* **sensors/** – fusão de sensores, VAD, análise de sinais.

  * **C files** – sensor fusion e VAD.
  * **src/** (Rust) – sensor filters e fusion logic.
* **gpu/** – abstração de GPU para CUDA, ROCm e Metal (dispatch, kernels, helpers).
* **navigation/** – path planner, free-space detector e obstacle avoider.
* **interaction/** – voice commands, feedback manager.

  * **src/** (Rust) – command parsing e feedback logic.
* **utils/** – logging, error handling, debug helpers.

  * **src/** (Rust) – error utils, debug helpers.
* **core\_build/** – scripts e CMakeLists para build core.
* **ffi/** – bindings e entrypoints C/Rust/C++.

---

#### 📂 Cargo.toml – configuração Rust workspace

* Lista crates, dependências e targets para subfolders Rust.

#### 📂 CMakeLists.txt – build system multi-linguagem

* Orquestra C/C++ e Rust, targets de GPU, testes e integração.

