# Segurança

API experimental para execução local. O servidor padrão escuta somente em `127.0.0.1`. Para qualquer acesso compartilhado, configure `FRUITVISION_API_KEY`, origem explícita, proxy com limites, TLS e controle de recursos. CORS não substitui autenticação.

Uploads têm limites de corpo, duração e pixels; arquivos e saídas usam nomes gerados pelo servidor e expiram. Esses controles não constituem sandbox de codecs ou modelos hostis. Use arquivos autorizados, pesos confiáveis e navegador/dependências atualizados.

Não publique chaves, arquivos pessoais, pesos privados ou saídas reais. Exemplos de configuração contêm campos vazios. O modo demo não realiza inferência nem inventa detecções. Detecção e segmentação 2D não certificam qualidade industrial ou segurança de alimentos.

Relate problemas com mídia sintética mínima, sem credenciais em issues públicas. Dependências/modelos mantêm suas próprias licenças; esta alteração não atribui nova licença ao código do projeto.
