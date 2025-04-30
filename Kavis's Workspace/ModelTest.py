import torch, itertools, warnings, math
from transformers import pipeline, GPT2LMHeadModel, GPT2TokenizerFast
from gramformer import Gramformer
from styleformer import Styleformer      # wrapper for prithivida/* Styleformer
from textwrap import fill
warnings.filterwarnings("ignore")

DEVICE   = 0 if torch.cuda.is_available() else -1

def pretty_show(results: dict, width: int = 100) -> None:
    """Nicely format the nested dict returned by professionalize(..., return_best=False)."""
    for stage, candidates in results.items():
        print(f"\n\033[1m{stage.upper()}\033[0m")     # bold heading
        for idx, sent in enumerate(candidates, 1):
            print(f"{idx:>2}. {fill(sent, width=width)}")

# ---------- Grammar-correction models ----------
pipes = {
    "t5_gec":  pipeline("text2text-generation",
                        model="vennify/t5-base-grammar-correction",
                        device=DEVICE),
    "bart_gec": pipeline("text2text-generation",
                         model="gotutiyan/gec-bart-large",
                         tokenizer="gotutiyan/gec-bart-large",
                         device=DEVICE)
}
gf = Gramformer(models=2, use_gpu = DEVICE != -1)   # only the corrector

# ---------- Formal-style models ----------
sf  = Styleformer(style = 0)                                   # casual ➜ formal
style_pipe = pipeline("text2text-generation",                  # alt. formalizer
                      model="rajistics/informal_formal_style_transfer",
                      device=DEVICE)

# ---------- Optional language-model scorer ----------
tok = GPT2TokenizerFast.from_pretrained("gpt2")
lm  = GPT2LMHeadModel.from_pretrained("gpt2").to(
         DEVICE if DEVICE!=-1 else "cpu").eval()
def ppl(text:str)->float:
    ids = tok(text,return_tensors="pt").input_ids.to(lm.device)
    with torch.no_grad(): loss = lm(ids,labels=ids).loss
    return math.exp(loss.item())

# ---------- Main utility -------------------------------------------------
def professionalize(text:str, n_candidates:int=2, return_best:bool=True):
    """
    Returns either:
      • one best professional rewrite   (return_best=True)   or
      • a dict {model:[cands]}          (return_best=False)
    """
    out = {}

    # ---- 1. Grammar correction -----------------------------------------
    corrected = []

    for name,pipe in pipes.items():
        gen = pipe(text, max_length=len(text.split())+48,
                   clean_up_tokenization_spaces=True)[0]["generated_text"]
        out.setdefault(name,[]).append(gen)
        corrected.append(gen)

    gf_iter = gf.correct(text, max_candidates=n_candidates)
    gf_cands = list(itertools.islice(gf_iter, n_candidates)) if gf_iter else [text]
    out["gramformer"] = gf_cands
    corrected += gf_cands

    # ---- 2. Style transfer (each corrected sentence ➜ formal) ----------
    formalized = []
    for sent in corrected:
        # Styleformer (fast)
        sf_out = sf.transfer(sent)
        if sf_out: formalized.append(sf_out[0])

        # Rajistics T5 style-transfer
        txt = style_pipe(sent, max_length=256,
                         clean_up_tokenization_spaces=True)[0]["generated_text"]
        formalized.append(txt)

    out["styleformer"]        = formalized[:len(corrected)]
    out["t5_style_transfer"]  = formalized[len(corrected):]

    # ---- 3. Pick the most fluent ---------------------------------------
    if return_best:
        scored = sorted({cand for lst in out.values() for cand in lst},
                        key=ppl)
        return scored[0]      # lowest perplexity
    else:
        return out
    
err = ("So uh, me and my friends we was like gonna went to the movies but hmm.. "
"nobody ain't had no car so we was just sitting there waiting for nobody to come get us, "
"and then Jimmy he say he seen the movie already but it weren't even out yet so we just decide to go eat some tacos"
" even though none of us don't like spicy and it was raining but we ain't bring no jackets or nothing so we just be wet"
" and cold the whole time.")

full = professionalize(err, return_best=False)
pretty_show(full)

print("\n\033[92mBEST (lowest GPT-2 perplexity)\033[0m\n",
      professionalize(err))