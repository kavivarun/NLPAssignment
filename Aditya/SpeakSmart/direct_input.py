# direct_input.py
import streamlit as st, time, utility
from Ollama_client import chat_with_models

class DirectText:
    """
    Paste-or-type text, run the same model pipeline used for audio uploads.
    Uses the CoEdit, Grammar+Style, Translation, and Ollama branches already
    wired in utility.py.
    """

    def __init__(self):
        # session keys so results persist across reruns
        for k in (
            "direct_raw", "direct_paraphrased", "direct_grammar",
            "direct_coedit", "direct_results"
        ):
            st.session_state.setdefault(k, None)

    # ---------- UI entry point ----------
    def enter_text(self):
        st.title("Direct Text Input")

        st.session_state.direct_raw = st.text_area(
            "Enter or paste the text you want to improve:",
            value=st.session_state.direct_raw or "",
            height=250,
            key="direct_text_area"
        )

        if not st.session_state.direct_raw:
            st.info("Add some text above to get started.")
            return

        processing_method = st.radio(
            "Choose processing method:",
            ("Parallel Processing (Faster)", "Sequential Processing (Original)"),
            key="direct_processing_method"
        )

        if st.button("Process Text"):
            if processing_method.startswith("Parallel"):
                self._run_parallel()
            else:
                self._run_sequential()

        # show results if we have them
        if st.session_state.direct_results:
            self._display_all_results(
                st.session_state.direct_raw,
                st.session_state.direct_results
            )

    # ---------- back-end wrappers ----------
    def _run_parallel(self):
        st.info("🚀 Running models in parallel…")
        prog  = st.progress(0)
        start = time.time()

        res = utility.process_text_parallel(
            st.session_state.direct_raw,
            progress_callback=lambda p: prog.progress(p/100)
        )
        st.session_state.direct_results = res

        # keep key intermediates for sequential-like access
        st.session_state.direct_paraphrased = res.get("style_improved")
        st.session_state.direct_grammar     = res.get("grammar_corrected")
        st.session_state.direct_coedit      = res.get("coedit_result")

        st.success(f"✨ Done in {time.time() - start:.1f}s")

    def _run_sequential(self):
        st.info("🐢 Running models sequentially…")

        # 1️⃣ grammar+style (existing helper)
        with st.spinner("Grammar & style models…"):
            para, gram = utility.paraphrase_single_text(st.session_state.direct_raw)
        st.session_state.direct_paraphrased = para
        st.session_state.direct_grammar     = gram

        # 2️⃣ CoEdit
        with st.spinner("Grammarly CoEdit…"):
            coedit, err = utility.run_coedit_model_async(st.session_state.direct_raw)
        if err:
            st.warning(f"CoEdit error – {err}")
        st.session_state.direct_coedit = coedit

        # 3️⃣ translation
        with st.spinner("Translation model…"):
            trans = utility.test_with_translation_models(st.session_state.direct_raw)

        # 4️⃣ Ollama / Mistral
        with st.spinner("Ollama (Mistral)…"):
            base = f"Please fix grammatical errors in this sentence and improve its style: {st.session_state.direct_raw}. Add it between `<fixg>` and `</fixg>` tags."
            ollama = chat_with_models(base, ["mistral:latest"])

        # wrap identical to parallel output so we can reuse the same renderer
        st.session_state.direct_results = {
            "grammar_corrected": gram,
            "style_improved":   para,
            "coedit_result":    coedit,
            "translation":      trans,
            "ollama_results":   ollama,
            "errors":           []
        }

        st.success("✨ Sequential pipeline complete.")

    # ---------- renderer ----------
    def _display_all_results(self, original, results):
        """
        Exactly the same visual block you already use for uploads,
        but without the download button (no filename here).
        """
        if results["errors"]:
            st.error("Some errors occurred:")
            for e in results["errors"]:
                st.write(f"• {e}")

        # Grammar + Style
        if results.get("style_improved"):
            st.markdown("## Grammar & Style Corrected")
            st.write(results["style_improved"])
            utility.display_text_quality_metrics(
                original, results["style_improved"], "Text Quality Metrics"
            )
            if results.get("grammar_corrected"):
                with st.expander("Intermediate grammar correction"):
                    st.write(results["grammar_corrected"])

        # CoEdit
        if results.get("coedit_result"):
            st.markdown("---")
            st.markdown("### CoEdit Result")
            st.write(results["coedit_result"])
            utility.display_text_quality_metrics(
                original, results["coedit_result"], "Text Quality Metrics: CoEdit"
            )

        # Translation
        if results.get("translation"):
            st.markdown("---")
            st.markdown("### Translation Model Result")
            st.write(results["translation"])
            utility.display_text_quality_metrics(
                original, results["translation"], "Text Quality Metrics: Translation"
            )

        # Ollama / Mistral
        if results.get("ollama_results"):
            extracted = utility.extract_from_model_responses(results["ollama_results"])
            if extracted.get("mistral:latest"):
                st.markdown("---")
                st.markdown("### Mistral Result")
                st.write(extracted["mistral:latest"])
                utility.display_text_quality_metrics(
                    original, extracted["mistral:latest"], "Text Quality Metrics: Mistral"
                )
