import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle, Send, Bot, User, Loader } from 'lucide-react';
import apiService from '../../services/api';

const Message = ({ message, isUser }) => (
  <div className={`flex items-start space-x-3 ${isUser ? 'flex-row-reverse space-x-reverse' : ''}`}>
    <div className={`p-2 rounded-lg ${isUser ? 'bg-blue-600' : 'bg-slate-700'}`}>
      {isUser ? <User className="w-5 h-5 text-white" /> : <Bot className="w-5 h-5 text-blue-400" />}
    </div>
    <div className={`flex-1 ${isUser ? 'text-right' : ''}`}>
      <div
        className={`inline-block px-4 py-2 rounded-lg ${
          isUser ? 'bg-blue-600 text-white' : 'bg-slate-700 text-slate-100'
        }`}
      >
        <p className="text-sm">{message.text}</p>
      </div>
      <p className="text-xs text-slate-500 mt-1">{message.timestamp}</p>
    </div>
  </div>
);

const Chatbot = () => {
  const [messages, setMessages] = useState([
    {
      text: "Hello! I'm Drishti AI Assistant. Ask me about crowd status, risk levels, or any alerts.",
      isUser: false,
      timestamp: new Date().toLocaleTimeString(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = {
      text: input,
      isUser: true,
      timestamp: new Date().toLocaleTimeString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await apiService.askQuestion(input);
      
      const botMessage = {
        text: response.answer,
        isUser: false,
        timestamp: new Date().toLocaleTimeString(),
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = {
        text: 'Sorry, I encountered an error. Please make sure the backend is running with Gemini API configured.',
        isUser: false,
        timestamp: new Date().toLocaleTimeString(),
      };

      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const quickQuestions = [
    'How many people are there?',
    'What is the current risk level?',
    'Is there any fire detected?',
    'What is the crowd density?',
  ];

  return (
    <div className="bg-slate-800 rounded-lg shadow-xl overflow-hidden border border-slate-700">
      {/* Header */}
      <div
        className="bg-slate-900 px-4 py-3 flex items-center justify-between cursor-pointer border-b border-slate-700"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center space-x-2">
          <MessageCircle className="w-5 h-5 text-blue-400" />
          <h2 className="text-lg font-semibold text-white">AI Assistant</h2>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
          <span className="text-xs text-slate-400">Online</span>
        </div>
      </div>

      {/* Chat Container */}
      {isExpanded && (
        <div className="h-96 flex flex-col">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((message, index) => (
              <Message key={index} message={message} isUser={message.isUser} />
            ))}
            {isLoading && (
              <div className="flex items-center space-x-2 text-slate-400">
                <Loader className="w-5 h-5 animate-spin" />
                <span className="text-sm">Thinking...</span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Quick Questions */}
          <div className="px-4 py-2 border-t border-slate-700 bg-slate-900/50">
            <p className="text-xs text-slate-400 mb-2">Quick questions:</p>
            <div className="flex flex-wrap gap-2">
              {quickQuestions.map((question, index) => (
                <button
                  key={index}
                  onClick={() => {
                    setInput(question);
                    setTimeout(() => handleSend(), 100);
                  }}
                  className="px-2 py-1 text-xs bg-slate-700 hover:bg-slate-600 text-slate-300 rounded transition-colors"
                  disabled={isLoading}
                >
                  {question}
                </button>
              ))}
            </div>
          </div>

          {/* Input */}
          <div className="p-4 border-t border-slate-700 bg-slate-900">
            <div className="flex items-center space-x-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask about crowd status, alerts, or safety..."
                className="flex-1 bg-slate-800 text-white px-4 py-2 rounded-lg border border-slate-600 focus:outline-none focus:border-blue-500 transition-colors"
                disabled={isLoading}
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || isLoading}
                className="bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 disabled:cursor-not-allowed text-white p-2 rounded-lg transition-colors"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Minimized State */}
      {!isExpanded && (
        <div className="px-4 py-3 text-center">
          <p className="text-sm text-slate-400">Click to open AI chat assistant</p>
        </div>
      )}
    </div>
  );
};

export default Chatbot;