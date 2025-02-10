from reasonflux import ReasonFlux

reasonflux = ReasonFlux(navigator_path='models',
                        template_matcher_path='/data/Research/yzc/models/jina-embeddings-v3',
                        inference_path='models',
                        template_path='data/template_library.json')
problem = """Given a sequence {aₙ} satisfying a₁=2, and aₙ₊₁=3aₙ+4 (n≥1), find the general term formula aₙ"""
reasonflux.reason(problem)