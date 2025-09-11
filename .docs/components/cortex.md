# Módulo Cortex

**Status:** 100% Completo

## Visão Geral

O Cortex é o cérebro central do sistema TrackieLLM. Ele é responsável por integrar todas as informações sensoriais, manter um modelo de mundo coeso, raciocinar sobre o contexto do usuário e tomar decisões proativas para garantir a segurança e a autonomia do usuário.

Esta versão final do Cortex representa a conclusão da visão do projeto, com um sistema de IA verdadeiramente proativo e consciente do contexto.

## Arquitetura de Raciocínio

O fluxo de raciocínio do Cortex foi aprimorado para fundir todas as fontes de dados em um único prompt dinâmico para o Large Language Model (LLM), garantindo que as informações mais críticas tenham a mais alta prioridade.

### Fontes de Contexto Integradas:

*   **Visão Computacional:** Detecção de objetos, análise de texto e estimativa de profundidade.
*   **Áudio Ambiente:** Detecção de sons críticos como alarmes de incêndio, sirenes e buzinas de carro.
*   **Dicas de Navegação:** Detecção de características do ambiente como degraus (para cima/baixo) e escadas.
*   **Estado do Usuário:** Detecção de movimento (parado, andando, correndo) e, criticamente, detecção de quedas.
*   **Histórico de Conversa:** Memória de curto prazo das últimas interações do usuário.
*   **Memória de Longo Prazo:** Fatos aprendidos sobre o usuário e o ambiente (ex: nome do usuário, locais familiares).

### Geração de Prompt Dinâmico

A função `generate_prompt_for_llm` em `reasoning.rs` agora constrói o prompt com a seguinte prioridade:

1.  **Alertas Críticos:** Se um alarme de incêndio ou uma queda for detectado, o prompt começa com "URGENTE:".
2.  **Perigos de Navegação:** Dicas como degraus ou escadas são adicionadas em seguida.
3.  **Contexto Geral:** O estado de movimento do usuário, objetos visíveis e informações da memória de longo prazo são incluídos.
4.  **Consulta do Usuário:** A pergunta direta do usuário é adicionada por último.

Este método garante que, mesmo que o usuário faça uma pergunta trivial, o LLM seja forçado a considerar primeiro os fatores de segurança mais importantes.

## Sistema de Decisão Proativa

O motor de decisão (`tk_decision_engine.c`) evoluiu de um simples executor de ações para um guardião de segurança proativo.

### Gatilhos de Ação Imediata

O motor agora pode enfileirar ações de emergência *antes* mesmo de consultar o LLM, com base em gatilhos de alta prioridade do contexto:

*   **Gatilho de Queda:** Se `user_motion_state == TK_MOTION_STATE_FALLING`, uma ação `EMERGENCY_ALERT` com a mensagem "Queda detectada! Você está bem?" é imediatamente enfileirada.
*   **Gatilho de Alarme de Incêndio:** Se `detected_sound_type == TK_AMBIENT_SOUND_FIRE_ALARM`, o sistema cancela todas as outras ações pendentes e enfileira um alerta de evacuação.

### Vetos de Segurança de Navegação

O motor de decisão agora tem o poder de vetar sugestões do LLM que possam ser perigosas:

*   Se o LLM sugere uma ação `NAVIGATE_GUIDE` (ex: "Vire à direita"), mas o `context_summary` indica que o caminho não está livre, a ação é invalidada.
*   Em vez de executar a ação perigosa, o motor a substitui por um aviso falado, como: "Eu ia sugerir virar à direita, mas detectei um obstáculo nesse caminho."

## Sistema de Memória de Longo Prazo

Para fornecer assistência personalizada, o Cortex agora possui uma memória de longo prazo persistente.

### Armazenamento e Persistência

*   A struct `LongTermMemory` em `memory_manager.rs` usa um `HashMap` para armazenar fatos chave-valor (ex: `"user_name": "Carlos"`).
*   As funções `save_memory_to_disk` e `load_memory_from_disk` serializam e desserializam o conteúdo da memória para um arquivo JSON, garantindo que as informações persistam entre as sessões.

### Aprendizado e Lembrança

*   **Aprendizado:** Ações `SYSTEM_SETTING` geradas pelo LLM (ex: após o usuário dizer "Lembre-se que meu nome é Carlos") são interceptadas pelo motor de decisão, que chama uma função FFI para atualizar o `MemoryManager` em Rust.
*   **Lembrança:** A função `generate_prompt_for_llm` inclui fatos relevantes da memória de longo prazo no prompt enviado ao LLM, fornecendo ao modelo o contexto necessário para personalizar suas respostas.
