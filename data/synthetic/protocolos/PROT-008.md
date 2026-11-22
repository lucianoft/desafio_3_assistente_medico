# PROT-008 - Politica de uso do assistente virtual medico

Versao: 1.0  |  Documento interno ficticio

## Escopo

O assistente apoia medicos com consulta a protocolos internos, resumo do prontuario e sugestao de proximos passos. Ele nao atende pacientes diretamente.

## Limites de atuacao

O assistente nunca prescreve medicamentos, nunca define doses e nunca da alta. Toda sugestao de conduta e marcada como pendente de validacao medica.

## Rastreabilidade

Cada interacao recebe um identificador de rastreio (trace_id) e e registrada no log de auditoria com pergunta, fontes consultadas, resposta e decisao do medico.

## Privacidade

O assistente acessa apenas dados minimos e pseudonimizados do paciente (codigo, idade, sexo, diagnosticos, exames). Nome, CPF e contato nao sao enviados ao modelo.
