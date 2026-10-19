# Déploiement k3s

## 1. Construire l'image en `linux/amd64`

Le nœud est en **amd64** (un Mac Apple Silicon produit de l'arm64). Depuis le Mac :

```sh
docker buildx build --platform linux/amd64 -t rag-project:latest --load .
```

Puis l'importer dans k3s :

```sh
docker save rag-project:latest | ssh root@10.10.20.19 "sudo k3s ctr images import -"
```

**Alternative** : construire directement sur le VPS (Docker/buildah), puis :

```sh
docker save rag-project:latest -o rag-project.tar
sudo k3s ctr images import rag-project.tar
```

Ou pousser sur un registry et adapter `image:` dans les manifestes.

## 2. Appliquer les manifestes

```sh
kubectl apply -f k8s/00-base.yaml
kubectl apply -f k8s/10-storage.yaml
kubectl apply -f k8s/40-ollama.yaml
kubectl apply -f k8s/20-api.yaml
kubectl apply -f k8s/30-dashboard.yaml
```

## 3. Tirer le modèle Ollama

```sh
kubectl -n rag delete job ollama-pull --ignore-not-found
kubectl apply -f k8s/jobs/ollama-pull.yaml
kubectl -n rag logs -f job/ollama-pull
```

### Variante : Ollama hors du cluster (poste GPU)

`40-ollama.yaml` fournit un Ollama dans le cluster. Pour utiliser plutôt un
poste GPU déjà équipé, il suffit de repointer l'API :

```sh
kubectl -n rag patch configmap rag-config --type merge \
  -p '{"data":{"OLLAMA_HOST":"http://<ip_du_poste>:11434"}}'
kubectl -n rag scale deployment/ollama --replicas=0
kubectl -n rag rollout restart deployment/api
```

Si le poste GPU n'est pas joignable depuis le nœud (autre réseau, tunnel),
`scripts/dev/ollama_relay.py` relaie un port local vers la cible :

```sh
python scripts/dev/ollama_relay.py --listen 0.0.0.0:11435 --target <ip_du_poste>:11434
# puis : OLLAMA_HOST=http://<ip_de_la_machine_relais>:11435
```

C'est un outil de développement : le relais doit rester lancé pour que l'API
atteigne le modèle.

## 4. Charger les documents

Le volume `rag-docs` démarre vide. Copie les documents dans le pod API :

```sh
POD=$(kubectl -n rag get pod -l app=api -o jsonpath='{.items[0].metadata.name}')
kubectl -n rag cp ./data/raw_docs/. "$POD:/app/data/raw_docs"

# relance l'ingestion (initContainer) pour intégrer les documents
kubectl -n rag rollout restart deployment/api
```

## Accès

- API : http://10.10.20.19:30080/docs
- Dashboard : http://10.10.20.19:30850

## Clé DeepSeek

```sh
kubectl -n rag create secret generic rag-secrets \
  --from-literal=DEEPSEEK_API_KEY=sk-... \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl -n rag rollout restart deployment/api
```

## Notes

- `local-path` : volumes stockés sur le disque du nœud (single node).
- L'ensemble est persistant : `rag-docs`, `rag-storage`, `rag-logs`, `ollama-models`.
- Les métriques d'évaluation vivent sur le serveur MLflow (`MLFLOW_TRACKING_URI`) et non dans un volume local.
- L'`initContainer` d'ingestion s'exécute à chaque démarrage de l'API : l'index reflète donc `data/raw_docs`.
- L'image Ollama pèse plusieurs Go ; le nœud dispose d'environ 20 Go d'espace éphémère.
