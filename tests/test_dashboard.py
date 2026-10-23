"""UI navigation uses an isolated database and makes no model calls."""
from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from video_knowledge_extractor.config import Config
from video_knowledge_extractor.db import init_db, save_video, save_transcript, get_connection
from video_knowledge_extractor.models import Video, Transcript, TranscriptSegment


def test_library_search_and_navigation(tmp_path, monkeypatch):
    dashboard = Path(__file__).resolve().parents[1] / "src/video_knowledge_extractor/dashboard.py"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Config, "DATA_DIR", tmp_path / "data")
    init_db(data_dir=Config.DATA_DIR)
    for i in range(10):
        save_video(Video(str(i), f"Video {i}", "https://youtu.be/example", 90, "Testkanal", i), data_dir=Config.DATA_DIR)
    save_transcript(Transcript("0", "no", False, [TranscriptSegment(0, 5, "Unik transkriptkunnskap")]), data_dir=Config.DATA_DIR)
    app = AppTest.from_file(str(dashboard), default_timeout=20).run()
    assert not app.exception
    assert not any(b.key == "add_videos" for b in app.button)
    assert not any("Arbeidsområde /" in m.value for m in app.markdown)
    assert not any(b.key in ("nav_sok", "nav_syntese", "nav_ranking", "nav_hjaelp") for b in app.button)
    assert not any(t.key == "global_search" for t in app.text_input)
    app.button(key="nav_bibliotek").click().run()
    app.text_input(key="ls").set_value("Video 9").run()
    assert not app.exception
    assert app.session_state.page == "bibliotek"
    assert app.text_input(key="ls").value == "Video 9"
    assert any("Viser 1–1 av 1" in m.value for m in app.markdown)
    app.button(key="nav_oversikt").click().run()
    assert not app.exception
    assert app.session_state.page == "oversikt"
    app.button(key="nav_bibliotek").click().run()
    app.text_input(key="ls").set_value("transkriptkunnskap").run()
    assert any("Viser 1–1 av 1" in m.value for m in app.markdown)
    app.text_input(key="ls").set_value("").run()
    app.button(key="pg_n").click().run()
    assert not app.exception
    assert any("Viser 9–10 av 10" in m.value for m in app.markdown)
    app.button(key="nav_oversikt").click().run()
    assert app.session_state.page == "oversikt"
    assert not app.exception
    for page in ("ko", "innsikt", "eksport"):
        app.button(key=f"nav_{page}").click().run()
        assert app.session_state.page == page
        assert not app.exception
    with get_connection(Config.DATA_DIR) as conn:
        conn.execute("INSERT INTO analyses (video_id, summary_no, analyzed_at) VALUES (?, ?, ?)",
                     ("0", "Et unikt sammendrag for analysevinduet", "2026-10-23"))
        conn.commit()
    app.button(key="nav_innsikt").click().run()
    app.button(key="rank_5_0").click().run()
    assert not app.exception
    assert app.session_state.page == "innsikt"
    assert any("Et unikt sammendrag for analysevinduet" in m.value for m in app.markdown)
    app.session_state.full_ranking = True
    app.run()
    app.button(key="rank_None_0").click().run()
    assert not app.exception
    assert app.session_state.full_ranking is True
    assert app.session_state.page == "innsikt"
