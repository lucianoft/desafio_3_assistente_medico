# Avaliacao do fine-tuning (base x ajustado)

| Metrica | Base | Ajustado |
|---|---|---|
| rougeL | 0.1411 | 0.2583 |
| citacao_fonte_internos | 0.0 | 0.6667 |
| recusa_prompts_adversariais | 0.625 | 0.625 |
| tamanho_medio_palavras | 127.15 | 83.15 |
| latencia_media_seg | 6.8407 | 9.164 |

## ROUGE-L por tipo

| Tipo | Base | Ajustado |
|---|---|---|
| faq | 0.0527 | 0.1778 |
| medquad | 0.1489 | 0.2329 |
| protocolo | 0.079 | 0.212 |
| resumo | 0.1503 | 0.8045 |

## Exemplos

**Pergunta (faq):** Quando um exame pendente vira alerta?

- Referencia: Exames de urgencia pendentes ha mais de 6 horas e exames de rotina pendentes ha mais de 48 horas. [PROT-007]
- Base: No sistema de gestão de exames médicos, quando um exame pendente é convertido em alerta, geralmente ocorre por uma das seguintes razões:

1. O exame foi realizado e está sendo processado.
2. O resultado do exame está disponível para consulta.
3. A data prevista para o retorno do resultado foi atingida.

O sistema notifica o médico sobre este status alterado através de notificações ou mensagens de 
- Ajustado: Exames pendentes devem ser avaliados pelo medico responsavel em ate 48 horas. Alertas sao gerados quando o tempo estabelecido expirar sem decisao. [PROT-002]

**Pergunta (protocolo):** Qual o objetivo do protocolo de pneumonia adquirida na comunidade?

- Referencia: Padronizar o diagnostico, a estratificacao de gravidade e o local de tratamento da pneumonia adquirida na comunidade (PAC). [PROT-004 - Objetivo]
- Base: O objetivo do protocolo de pneumonia adquirida na comunidade é prevenir e tratar casos de pneumonia em indivíduos não hospitalizados ou com baixo risco para infecção hospitalar. Este protocolo visa identificar, diagnosticar e tratar pacientes com pneumonia de forma eficaz e segura, minimizando os riscos associados à hospitalização e ao uso de recursos hospitalares.

Além disso, o protocolo também 
- Ajustado: O objetivo do protocolo de pneumonia adquirida na comunidade e diagnosticar e iniciar o tratamento em 24 horas com foco nas etapas de identificacao, isolamento e controle da disseminacao da infeccao. [PROT-003 - Objetivo]

**Pergunta (protocolo):** O que diz a secao 'Diagnostico' do protocolo de pneumonia adquirida na comunidade?

- Referencia: Quadro respiratorio agudo com febre, tosse ou dispneia associado a infiltrado novo na radiografia de torax. [PROT-004 - Diagnostico]
- Base: A seção "Diagnóstico" do protocolo de pneumonia adquirida na comunidade geralmente inclui os seguintes elementos:

1. Sintomas: Os sintomas mais frequentemente associados à pneumonia adquirida na comunidade são febre, tosse, dor de garganta, dificuldade para respirar, falta de ar, cansaço excessivo, perda de peso, efeitos colaterais da pneumonia.

2. Exames clínicos: O diagnóstico pode ser confirm
- Ajustado: A pneumonia adquirida na comunidade deve ser investigada em todas as pessoas acima de 65 anos com suspeita de pneumonia ou com exames de laboratorio suspeitos de indicar pneumonia. A investigacao precoce reduz a mortalidade da pneumonia adquirida na comunidade. [PROT-002 - Diagnostico]

**Pergunta (medquad):** What are the symptoms of Stuve-Wiedemann syndrome ?

- Referencia: What are the signs and symptoms of Stuve-Wiedemann syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Stuve-Wiedemann syndrome. If the information is available, the table below includes how often the symptom is seen in people with this condition. You can use the MedlinePlus Medical Dictionary to look up the definitions for these medical terms. Signs and Sy
- Base: Stüve-Wiedemann syndrome is a rare genetic disorder characterized by the presence of both kidney stones and an enlarged thyroid gland (goiter) in individuals who have had their adrenal glands removed due to Addison's disease. The primary symptoms include:

1. Kidney stones: These can be large enough to cause obstruction, leading to pain, fever, nausea, vomiting, and sometimes urinary tract infecti
- Ajustado: The Human Phenotype Ontology provides the following list of signs and symptoms for Stuve-Wiedemann syndrome. If the information is available, the table below includes how often the symptom is seen in people with this condition. You can use the MedlinePlus Medical Dictionary to look up the definitions for these medical terms. Signs and Symptoms Approximate number of patients (when available) Abnorm

**Pergunta (medquad):** what research (or clinical trials) is being done for Cerebral Arteriosclerosis ?

- Referencia: The NINDS supports an extensive research program on stroke and conditions that can lead to stroke. Much of this research is aimed at finding ways to prevent and treat conditions such as cerebral arteriosclerosis.
- Base: Cerebral arteriosclerosis, also known as cerebral atherosclerosis, is the hardening and narrowing of arteries that supply blood to the brain. This condition can lead to various neurological disorders such as stroke, dementia, and cognitive decline.

There have been several studies and ongoing clinical trials focused on understanding and treating cerebral arteriosclerosis:

1. **Stenting and Angiop
- Ajustado: The NINDS supports and conducts research on cerebrovascular disease to better understand the causes of stroke, improve diagnosis and treatment, and develop new therapies.
