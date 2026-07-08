import streamlit as st

SOCIAL_LINKS = {
    "Email": "mailto:work.mohitjoshi@gmail.com",
    "GitHub": "https://github.com/work-mohit",
    "LinkedIn": "https://linkedin.com/in/workmohitjoshi",
    "Instagram": "https://instagram.com/nothingbutmohitjoshi",
    "Twitter / X": "https://twitter.com/mohitjoshi__",
}


_FOOTER_ICONS = {
    "Email": '<path d="M2 4h20v16H2z" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M3 5l9 7 9-7" fill="none" stroke="currentColor" stroke-width="1.6"/>',
    "GitHub": '<path fill="currentColor" d="M12 1a11 11 0 0 0-3.48 21.44c.55.1.75-.24.75-.53v-1.85c-3.07.67-3.72-1.48-3.72-1.48-.5-1.28-1.23-1.62-1.23-1.62-1-.7.08-.68.08-.68 1.1.08 1.68 1.14 1.68 1.14.98 1.68 2.58 1.2 3.2.92.1-.71.38-1.2.7-1.48-2.45-.28-5.02-1.23-5.02-5.46 0-1.2.43-2.19 1.13-2.96-.11-.28-.49-1.4.11-2.92 0 0 .92-.3 3.03 1.13a10.4 10.4 0 0 1 5.52 0c2.1-1.43 3.02-1.13 3.02-1.13.6 1.52.22 2.64.11 2.92.71.77 1.13 1.76 1.13 2.96 0 4.24-2.58 5.17-5.04 5.44.39.34.75 1.02.75 2.06v3.05c0 .29.2.64.76.53A11 11 0 0 0 12 1z"/>',
    "LinkedIn": '<rect x="2" y="2" width="20" height="20" rx="2" fill="none" stroke="currentColor" stroke-width="1.6"/><path fill="currentColor" d="M7.2 9.5h2.4V17H7.2zM8.4 6a1.4 1.4 0 1 1 0 2.8 1.4 1.4 0 0 1 0-2.8zM11.6 9.5h2.3v1.03c.33-.6 1.14-1.23 2.34-1.23 2.5 0 2.96 1.65 2.96 3.79V17h-2.4v-3.5c0-.83-.02-1.9-1.16-1.9-1.16 0-1.34.9-1.34 1.84V17h-2.4z"/>',
    "Instagram": '<rect x="2.5" y="2.5" width="19" height="19" rx="5" fill="none" stroke="currentColor" stroke-width="1.6"/><circle cx="12" cy="12" r="4.2" fill="none" stroke="currentColor" stroke-width="1.6"/><circle cx="17.3" cy="6.7" r="1.1" fill="currentColor"/>',
    "Twitter / X": '<path fill="currentColor" d="M3 3l7.5 9.5L3.3 21H6l5.6-6.4L15.8 21H21l-8-10.1L20.4 3h-2.7l-5.1 5.9L8.3 3z"/>',
}


def render_footer():
    """
    Renders the icon-row footer + credit line. Reads from SOCIAL_LINKS above
    -- edit the URLs there. Any entry left as "" is skipped automatically
    instead of rendering a dead link.
    """
    icons_html = ""
    for label, url in SOCIAL_LINKS.items():
        if not url:
            continue
        svg_path = _FOOTER_ICONS.get(label, "")
        icons_html += f"""
        <a href="{url}" target="_blank" title="{label}" style="
            display:inline-flex; align-items:center; justify-content:center;
            width:36px; height:36px; margin:0 6px; border-radius:50%;
            color:inherit; text-decoration:none; opacity:0.75;
            transition:opacity 0.15s ease;"
           onmouseover="this.style.opacity=1" onmouseout="this.style.opacity=0.75">
            <svg width="20" height="20" viewBox="0 0 24 24">{svg_path}</svg>
        </a>"""

    st.markdown(
        f"""
        <div style="text-align:center; padding-top:0.5rem;">
            <div>{icons_html}</div>
            <div style="font-size:0.8rem; opacity:0.6; margin-top:0.4rem;">
                AI App created by Mohit Joshi
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
