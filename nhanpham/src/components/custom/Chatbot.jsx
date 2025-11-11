import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User } from 'lucide-react';

export const Chatbot = ({ patientName }) => {
    const [messages, setMessages] = useState([
        {
            id: 1,
            type: 'bot',
            text: `Hello! I'm your AI medical assistant. How can I help you with ${patientName}'s case today?`,
            timestamp: new Date()
        }
    ]);
    const [inputMessage, setInputMessage] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSendMessage = (e) => {
        e.preventDefault();

        if (inputMessage.trim() === '') return;

        // Add user message
        const userMessage = {
            id: messages.length + 1,
            type: 'user',
            text: inputMessage,
            timestamp: new Date()
        };

        setMessages([...messages, userMessage]);
        setInputMessage('');
        setIsTyping(true);

        // Simulate bot response
        setTimeout(() => {
            const botMessage = {
                id: messages.length + 2,
                type: 'bot',
                text: `I understand you're asking about "${inputMessage}". As an AI assistant, I can help analyze patient data, suggest diagnostic approaches, and provide medical information. How would you like to proceed?`,
                timestamp: new Date()
            };
            setMessages(prev => [...prev, botMessage]);
            setIsTyping(false);
        }, 1500);
    };

    return (
        <div className="flex flex-col h-full bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl overflow-hidden">
            {/* Header */}
            <div className="bg-teal-500/20 border-b border-teal-500/30 px-6 py-4">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-teal-500 rounded-full flex items-center justify-center">
                        <Bot className="w-6 h-6 text-white" />
                    </div>
                    <div>
                        <h3 className="font-semibold text-white">AI Medical Assistant</h3>
                        <p className="text-xs text-teal-300">Online • Ready to help</p>
                    </div>
                </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-6 space-y-4" style={{ maxHeight: '500px' }}>
                {messages.map((message) => (
                    <div
                        key={message.id}
                        className={`flex gap-3 ${message.type === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
                    >
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${message.type === 'bot' ? 'bg-teal-500/20' : 'bg-blue-500/20'
                            }`}>
                            {message.type === 'bot' ? (
                                <Bot className="w-5 h-5 text-teal-400" />
                            ) : (
                                <User className="w-5 h-5 text-blue-400" />
                            )}
                        </div>
                        <div className={`flex-1 ${message.type === 'user' ? 'text-right' : 'text-left'}`}>
                            <div className={`inline-block max-w-[80%] px-4 py-3 rounded-lg ${message.type === 'bot'
                                    ? 'bg-white/10 text-white'
                                    : 'bg-teal-500/30 text-white'
                                }`}>
                                <p className="text-sm">{message.text}</p>
                            </div>
                            <p className="text-xs text-gray-500 mt-1">
                                {message.timestamp.toLocaleTimeString()}
                            </p>
                        </div>
                    </div>
                ))}

                {isTyping && (
                    <div className="flex gap-3">
                        <div className="w-8 h-8 bg-teal-500/20 rounded-full flex items-center justify-center">
                            <Bot className="w-5 h-5 text-teal-400" />
                        </div>
                        <div className="bg-white/10 px-4 py-3 rounded-lg">
                            <div className="flex gap-1">
                                <span className="w-2 h-2 bg-teal-400 rounded-full animate-bounce"></span>
                                <span className="w-2 h-2 bg-teal-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></span>
                                <span className="w-2 h-2 bg-teal-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></span>
                            </div>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <form onSubmit={handleSendMessage} className="border-t border-white/10 p-4">
                <div className="flex gap-2">
                    <input
                        type="text"
                        value={inputMessage}
                        onChange={(e) => setInputMessage(e.target.value)}
                        placeholder="Type your message..."
                        className="flex-1 bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-teal-500 focus:bg-white/10 transition-all"
                    />
                    <button
                        type="submit"
                        disabled={inputMessage.trim() === ''}
                        className="bg-teal-500 hover:bg-teal-600 disabled:bg-gray-600 disabled:cursor-not-allowed text-white px-6 py-3 rounded-lg transition-all flex items-center gap-2"
                    >
                        <Send className="w-5 h-5" />
                    </button>
                </div>
            </form>
        </div>
    );
};
