import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css'; // Important for text layer
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';

pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;

// TODO: Import child components (ChatMessage, ChatWindow, PDFViewer, SectionTree, TextSearch, SearchResultsDisplay)

const ChatMessage = ({ sender, text, citations, onReferenceClick }) => (
    <div className={`mb-2 flex ${sender === 'user' ? 'justify-end' : 'justify-start'}`}>
      <div className={`${sender === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-800'} rounded-2xl p-3 max-w-lg shadow whitespace-pre-wrap`}>
        <p>{text}</p>
        {sender === 'assistant' && citations && citations.length > 0 && (
          <div className="mt-3 pt-2 border-t border-gray-300">
            <p className="text-sm font-semibold mb-1 text-gray-700">References:</p>
            <ul className="list-none pl-0 text-sm">
              {citations.map((cite, idx) => (
                <li key={cite.chunk_id || idx} className="mb-1">
                  <button
                    onClick={() => onReferenceClick(cite)}
                    className="text-indigo-600 hover:text-indigo-800 hover:underline focus:outline-none text-left"
                    title={`Chunk ID: ${cite.chunk_id}\nPage: ${cite.start_page}`}
                  >
                    {cite.title || `Reference on Page ${cite.start_page}`} (ID: ...{cite.chunk_id ? cite.chunk_id.slice(-6) : 'N/A'})
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );

const ChatWindow = ({ messages, onSend, onReferenceClick }) => {
    const [input, setInput] = useState('');
    const bottomRef = useRef(null);
  
    useEffect(() => {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);
  
    const handleSend = () => {
      if (!input.trim()) return;
      onSend(input.trim());
      setInput('');
    };
  
    return (
      <div className="flex flex-col h-full bg-white shadow rounded">
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {messages.map((msg, idx) => (
            <ChatMessage
                key={idx}
                sender={msg.sender}
                text={msg.text}
                citations={msg.citations}
                onReferenceClick={onReferenceClick} />
          ))}
          <div ref={bottomRef} />
        </div>
        <div className="p-4 border-t flex">
          <input
            className="flex-1 border rounded-l-lg p-2 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask your question about the bill..."
          />
          <button
            className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-r-lg"
            onClick={handleSend}
          >
            Send
          </button>
        </div>
      </div>
    );
  };

const PDFViewer = ({ fileUrl, highlights, onPageClick, scrollToPageNum, activeChunkId }) => {
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [numPages, setNumPages] = useState(null);
    const pageRefs = useRef([]);
  
    useEffect(() => {
      if (numPages) {
        pageRefs.current = Array(numPages).fill(null).map((_, i) => pageRefs.current[i] || React.createRef());
      }
    }, [numPages]);
  
    useEffect(() => {
      if (scrollToPageNum && pageRefs.current[scrollToPageNum - 1]?.current) {
        pageRefs.current[scrollToPageNum - 1].current.scrollIntoView({
          behavior: 'smooth',
          block: 'start',
        });
      }
    }, [scrollToPageNum]);
  
    const onDocumentLoadSuccess = ({ numPages: nextNumPages }) => {
      setNumPages(nextNumPages);
      setLoading(false);
    };
  
    return (
      <div className="relative h-full overflow-y-auto border border-gray-300 rounded bg-gray-50">
        {loading && <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-75 z-10">Loading PDF...</div>}
        {error && <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-75 text-red-600 z-10">Error loading PDF: {error}</div>}
        <Document
          file={fileUrl}
          onLoadError={(err) => { setError(err.message || 'Failed to load PDF'); setLoading(false); }}
          onLoadSuccess={onDocumentLoadSuccess}
        >
          {numPages && Array.from(new Array(numPages), (el, index) => (
            <div
              key={`page_container_${index + 1}`}
              ref={pageRefs.current[index]}
              className={`relative pdf-page-container border-b border-gray-200 ${(scrollToPageNum === (index + 1) && activeChunkId) ? 'ring-2 ring-indigo-500 outline-none' : ''}`}
              onClick={() => onPageClick && onPageClick(index + 1)}
            >
              <Page 
                pageNumber={index + 1} 
                width={600} // Consider making this dynamic
                renderTextLayer={true} // Important for text selection/searchability within PDF
                renderAnnotationLayer={true}
              />
              {highlights && highlights.items && highlights.items
                .filter((h) => h.page === index + 1)
                .map((h, hIdx) => (
                  <div
                    key={`highlight_${hIdx}_${h.page}`}
                    className="absolute bg-yellow-400 opacity-50 pointer-events-none"
                    style={{
                      top: `${h.y}%`, left: `${h.x}%`,
                      width: `${h.width}%`, height: `${h.height}%`,
                    }}
                  />
                ))}
            </div>
          ))}
        </Document>
      </div>
    );
  };


// Section tree navigator component
const SectionTree = ({ sections, onSelect, currentSectionId }) => (
    <div className="p-2 overflow-y-auto border-r h-full bg-gray-50">
      <h3 className="text-lg font-semibold mb-2 text-gray-700">Document Sections</h3>
      <ul>
        {sections.map((sec) => (
          <li key={sec.id} className="mb-1">
            <button
              className={`text-left w-full p-1 rounded hover:bg-indigo-100 ${currentSectionId === sec.id ? 'bg-indigo-200 font-semibold' : ''}`}
              onClick={() => onSelect(sec)}
            >
              {`Page No. ${sec.page}: `}
              {sec.title}
            </button>
            {sec.subsections && sec.subsections.length > 0 && (
              <ul className="pl-4 mt-1">
                {sec.subsections.map((sub) => (
                  <li key={sub.id} className="mb-1">
                    <button
                      className={`text-left w-full p-1 rounded hover:bg-indigo-100 text-sm ${currentSectionId === sub.id ? 'bg-indigo-200 font-semibold' : ''}`}
                      onClick={() => onSelect(sub)}
                    >
                      {`Page No. ${sub.page}: `}
                      {sub.title}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
  
const TextSearch = ({ onSearch, isLoading }) => {
    const [searchTerm, setSearchTerm] = useState('');
  
    const handleSearch = (e) => {
      e.preventDefault();
      if (!searchTerm.trim()) return;
      onSearch(searchTerm.trim());
    };
  
    return (
      <form onSubmit={handleSearch} className="p-3 border-b bg-gray-50">
        <label htmlFor="text-search-input" className="block text-sm font-medium text-gray-700 mb-1">
            Text Search Document
        </label>
        <div className="flex">
          <input
            id="text-search-input"
            type="text"
            className="flex-1 border border-gray-300 rounded-l-md p-2 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Enter keywords..."
            disabled={isLoading}
          />
          <button
            type="submit"
            className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-r-md disabled:opacity-50"
            disabled={isLoading}
          >
            {isLoading ? 'Searching...' : 'Search'}
          </button>
        </div>
      </form>
    );
  };
  
const SearchResultsDisplay = ({ results, onItemClick, isLoading }) => {
    if (isLoading) {
        return <div className="p-3 text-center text-gray-500">Searching for results...</div>;
    }
    if (!results || results.length === 0) {
      return <div className="p-3 text-center text-gray-500">No search results found.</div>;
    }
    return (
      <div className="p-2 overflow-y-auto flex-1"> {/* flex-1 to take available space */}
        <h4 className="text-md font-semibold mb-2 text-gray-700">Search Results:</h4>
        <ul className="space-y-2">
          {results.map((item) => (
            <li key={item.chunk_id} className="border rounded-md hover:bg-indigo-50 transition-colors">
              <button
                onClick={() => onItemClick(item)}
                className="text-left w-full p-3 focus:outline-none"
              >
                <strong className="text-indigo-700 block mb-1">{item.title || `Page ${item.start_page}`}</strong>
                <p className="text-sm text-gray-600 leading-relaxed">{item.snippet || item.text.substring(0, 200)}...</p>
                <span className="text-xs text-gray-500 mt-1 block">Page: {item.start_page} (ID: ...{item.chunk_id.slice(-6)})</span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    );
  };


// Main App component
export default function App() {
  const [messages, setMessages] = useState([{sender: 'assistant', text: "Hello! How can I help you with the spending bill today?"}]);
  const [highlights, setHighlights] = useState({ items: [] }); // Only items needed based on PDFViewer
  const [sections, setSections] = useState([]);
  const [fileUrl, setFileUrl] = useState('/2024_further_consolidated_appropriations_act.pdf'); // Ensure this PDF is in backend static folder
  const [scrollToPageNum, setScrollToPageNum] = useState(null);
  const [activeChunkId, setActiveChunkId] = useState(null); // For visual cue on active chunk's page

  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isQueryingLLM, setIsQueryingLLM] = useState(false);
  const [currentSectionId, setCurrentSectionId] = useState(null);

  // Load table of contents on mount
  useEffect(() => {
    fetch('http://localhost:8000/api/sections')
      .then((res) => res.ok ? res.json() : Promise.reject(`API Error: ${res.status}`))
      .then((data) => setSections(data))
      .catch(error => console.error("Failed to load sections:", error));
  }, []);

  const fetchHighlightCoordinates = async (chunkId) => {
    try {
      const res = await fetch('http://localhost:8000/api/highlight-coordinates', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chunk_ids: [chunkId] }),
      })
      if (!res.ok) throw new Error(`Highlight API error: ${res.status}`);
      const data = await res.json(); // Expects { items: [...] }
      setHighlights(data);
    } catch (error) {
      console.error("Failed to fetch highlight coordinates:", error);
      setHighlights({ items: [] }); // Clear highlights on error
    }
  };

  const handleReferenceClick = useCallback(async (citation) => {
    console.log("Clicked reference:", citation);
    setScrollToPageNum(citation.start_page);
    setActiveChunkId(citation.chunk_id);
    setCurrentSectionId(null); // Clear section selection
    // Fetch precise highlights for this specific chunk
    // This is a "stretch goal" part. If /api/highlight-coordinates exists:
    await fetchHighlightCoordinates(citation.chunk_id);
    // If not, the page will scroll and have a general 'active' ring.
  }, []);

  const handleSend = async (text) => {
    const userMsg = { sender: 'user', text };
    setMessages((prev) => [...prev, userMsg]);
    setIsQueryingLLM(true);
    setActiveChunkId(null); // Clear previous active chunk highlights

    try {
      const res = await fetch('http://localhost:8000/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: text }),
      });
      if (!res.ok) throw new Error(`LLM Query API error: ${res.status}`);
      const { answer, citations } = await res.json();

      const assistantMsg = { sender: 'assistant', text: answer, citations: citations || [] };
      setMessages((prev) => [...prev, assistantMsg]);

      // Optionally, highlight all context citations immediately
      if (citations && citations.length > 0) {
        // If we want to highlight all citations returned with the answer:
        // const chunkIds = citations.map(c => c.chunk_id);
        // const highlightRes = await fetch(`/api/highlight-coordinates`, { // Assuming batch endpoint
        //   method: 'POST',
        //   headers: { 'Content-Type': 'application/json' },
        //   body: JSON.stringify({ chunk_ids: chunkIds }),
        // });
        // if (highlightRes.ok) setHighlights(await highlightRes.json());
        // For now, let's clear highlights until a specific reference is clicked.
        setHighlights({ items: [] });
      } else {
        setHighlights({ items: [] });
      }

    } catch (error) {
      console.error("Failed to query LLM:", error);
      const errorMsg = { sender: 'assistant', text: "Sorry, I encountered an error trying to answer your question." };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
        setIsQueryingLLM(false);
    }
  };

  const handleSectionSelect = useCallback((section) => {
    console.log("Selected section:", section);
    if (section.page) {
      setScrollToPageNum(section.page);
    }
    setActiveChunkId(null); // Clear specific chunk highlight
    setHighlights({ items: [] }); // Clear precise highlights
    setCurrentSectionId(section.id);
  }, []);

  const handleTextSearch = async (searchTerm) => {
    setIsSearching(true);
    setSearchResults([]);
    setActiveChunkId(null);
    try {
      const res = await fetch(`/api/search?query=${encodeURIComponent(searchTerm)}`); // Assuming GET, or adapt to POST
      if (!res.ok) throw new Error(`Search API error: ${res.status}`);
      const { results } = await res.json();
      setSearchResults(results || []);
    } catch (error) {
      console.error("Failed to perform text search:", error);
      setSearchResults([]); // Clear results on error
    } finally {
      setIsSearching(false);
    }
  };

  const handleSearchResultClick = useCallback(async (searchItem) => {
    console.log("Clicked search result:", searchItem);
    setScrollToPageNum(searchItem.start_page);
    setActiveChunkId(searchItem.chunk_id);
    setCurrentSectionId(null);
    // Fetch precise highlights for this specific search item's chunk
    await fetchHighlightCoordinates(searchItem.chunk_id);
  }, []);


  return (
    <div className="flex flex-col h-screen bg-gray-100">
      {/* ─── HEADER ───────────────────────────── */}
      <header className="bg-white p-4 shadow flex items-center">
        <img
          src="/cartoon_bill_medium.png"
          alt="Bill illustration"
          className="h-16 w-auto mr-4"
        />
        <h1 className="text-2xl font-bold text-gray-800">
          Ask Me About the Spending Omnibus!
        </h1>
      </header>

      {/* ─── MAIN 3-PANE ───────────────────────── */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Pane */}
        <aside className="w-1/4 bg-gray-50 border-r border-gray-300 flex flex-col overflow-y-auto">
          <TextSearch onSearch={handleTextSearch} isLoading={isSearching} />
          {isSearching || searchResults.length > 0 ? (
            <SearchResultsDisplay
              results={searchResults}
              onItemClick={handleSearchResultClick}
              isLoading={isSearching}
            />
          ) : (
            <SectionTree
              sections={sections}
              onSelect={handleSectionSelect}
              currentSectionId={currentSectionId}
            />
          )}
        </aside>

        {/* Middle Pane */}
        <main className="lex-1 flex flex-col p-4 overflow-hidden">
          <PDFViewer
            fileUrl={fileUrl}
            highlights={highlights}
            scrollToPageNum={scrollToPageNum}
            activeChunkId={activeChunkId}
          />
        </main>

        {/* Right Pane */}
        <section className="w-1/4 p-4 border-l border-gray-300 flex flex-col overflow-y-auto">
          <ChatWindow
            messages={messages}
            onSend={handleSend}
            onReferenceClick={handleReferenceClick}
          />
        </section>
      </div>
    </div>
  );
}