$(document).ready(function () {

    // Initialization
    var botui = new BotUI('hello-world');

    // Declare conversation function
    function conversation(firstname, lastname, genre, model) {
        var stop = false;
        botui.action.text({
            action: {
                placeholder: ''
            }
        }).then(function (res) {
            if (res.value == 'goodbye' || res.value == 'evaluate') {
                stop = true;
                return false;
            }
            else {
                botui.message.add({
                    loading: true
                }).then(function (index) {
                    if (model == 'A') {
                        ajaxUrl = '/ajax/chat_rnn/'
                    }
                    else {
                        ajaxUrl = '/ajax/chat_ngram/'
                    }
                    console.log(model);
                    console.log(ajaxUrl)
                    $.ajax({
                        url: ajaxUrl,
                        data: {
                            'userinput': res.value
                        },
                        dataType: 'json',
                        success: function (data) {
                            botui.message.update(index, {
                                loading: false,
                                content: data,
                            })
                        },
                    })
                });
            }
        }).then(function () {
            // user would like to keep talking
            if (stop == false) {
                conversation(firstname, lastname, genre, model)
            }
            // user would like to stop talking
            else {
                botui.message.add({
                    content: 'Talk to you later!'
                })
                evaluate(firstname, lastname, genre, model)
            }
        })
    }


    function evaluate(firstname, lastname, genre, model) {
        botui.message.add({
            content: "What a fun chat! Let's evaluate how I did."
        }).then(function () {
            botui.message.add({
                content: "How would you describe my grammar (syntax)?"
            }).then(function () {
                return botui.action.button({
                    delay: 600,
                    cssClass: 'likert',
                    action: [
                        { text: 'really bad', value: 5 },
                        { text: 'bad', value: 4 },
                        { text: 'okay', value: 3 },
                        { text: 'good', value: 2 },
                        { text: 'really good', value: 1 },
                    ]
                }).then(function (res) {
                    var syntactic_score = res.value;
                    botui.message.add({
                        content: "How would you describe my understanding of the context of our conversation (semantics)?"
                    }).then(function () {
                        return botui.action.button({
                            delay: 600,
                            cssClass: 'likert',
                            action: [
                                { text: 'really bad', value: 5 },
                                { text: 'bad', value: 4 },
                                { text: 'okay', value: 3 },
                                { text: 'good', value: 2 },
                                { text: 'really good', value: 1 },
                            ]
                        }).then(function (res) {
                            var semantic_score = res.value;
                            botui.message.add({
                                content: "Was I fun to talk to?"
                            }).then(function () {
                                return botui.action.button({
                                    delay: 600,
                                    cssClass: 'likert',
                                    action: [
                                        { text: 'really boring', value: 5 },
                                        { text: 'boring', value: 4 },
                                        { text: 'okay', value: 3 },
                                        { text: 'fun', value: 2 },
                                        { text: 'really fun', value: 1 },
                                    ]
                                }).then(function (res) {
                                    var fun_score = res.value;
                                    botui.message.add({
                                        content: "Did I sound as if I were in a " + genre + " movie?"
                                    }).then(function () {
                                        return botui.action.button({
                                            delay: 600,
                                            cssClass: 'likert',
                                            action: [
                                                { text: 'never', value: 5 },
                                                { text: 'barely', value: 4 },
                                                { text: 'unsure', value: 3 },
                                                { text: 'sometimes', value: 2 },
                                                { text: 'a lot', value: 1 },
                                            ]
                                        }).then(function (res) {
                                            var genre_score = res.value;
                                            var user_data = [{ firstname },
                                            { lastname },
                                            { genre },
                                            { model },
                                            { syntactic_score },
                                            { semantic_score },
                                            { fun_score },
                                            { genre_score },]
                                            console.log(user_data)
                                            $.ajax({
                                                url: '/ajax/write_eval/',
                                                data: {
                                                    'usereval': JSON.stringify(user_data)
                                                },
                                                dataType: 'json',
                                                success: function (data) {
                                                    botui.message.add({
                                                        content: data
                                                    })
                                                    continueChat()
                                                },
                                            })
                                        })
                                    })
                                })
                            })
                        })
                    })
                })
            })
        })
    };

    function userevaluation() {
        botui.message.add({
            content: "Thanks for helping test my abilities! What is your first name?"
        }).then(function () {
            return botui.action.text({
                action: {
                    placeholder: 'first name'
                }
            }).then(function (res) {
                var firstname = res.value
                botui.message.add({
                    content: "Hi " + firstname + "! What is your last name?"
                }).then(function () {
                    return botui.action.text({
                        action: {
                            placeholder: 'last name'
                        }
                    }).then(function (res) {
                        var lastname = res.value;
                        botui.message.add({
                            content: res.value + ", that's a cool last name!"
                        }).then(function () {
                            botui.message.add({
                                content: "Which movie genre would you like to evaluate?"
                            })
                        }).then(function () {
                            return botui.action.select({
                                action: {
                                    placeholder: "Select Genre",
                                    value: 'comedy',
                                    searchselect: true, // Default: true, false for standart dropdown
                                    label: 'text', // dropdown label variable
                                    options: [
                                        { value: "action", text: "Action" },
                                        { value: "adventure", text: "Adventure" },
                                        { value: "animation", text: "Animation" },
                                        { value: "biography", text: "Biography" },
                                        { value: "comedy", text: "Comedy" },
                                        { value: "crime", text: "Crime" },
                                        { value: "documentary", text: "Documentary" },
                                        { value: "drama", text: "Drama" },
                                        { value: "family", text: "Family" },
                                        { value: "fantasy", text: "Fantasy" },
                                        { value: "film-noir", text: "Film-noir" },
                                        { value: "horror", text: "Horror" },
                                        { value: "music", text: "Music" },
                                        { value: "musical", text: "Musical" },
                                        { value: "mystery", text: "Mystery" },
                                        { value: "romance", text: "Romance" },
                                        { value: "sci-fi", text: "Sci-fi" },
                                        { value: "short", text: "Short" },
                                        { value: "sport", text: "Sport" },
                                        { value: "thriller", text: "Thriller" },
                                        { value: "war", text: "War" },
                                        { value: "western", text: "Western" },
                                    ],
                                    button: {
                                        icon: 'check',
                                        label: 'OK'
                                    }
                                }
                            })
                        }).then(function (res) {
                            var genre = res.value
                            botui.message.add({
                                content: `Great, I am going to speak as if I'm in a ` + res.value + ` film.`
                            }).then(function () {
                                return botui.message.add({
                                    content: "Ready to chat?!?"
                                })
                            }).then(function () {
                                return botui.action.button({
                                    action: [
                                        {
                                            text: "Let\'s do this!",
                                            value: 'yes'
                                        },
                                        {
                                            text: 'No',
                                            value: 'no'
                                        }
                                    ]
                                }).then(function (res) {
                                    if (res.value == 'yes') {
                                        var myModels = ['A', 'B'];
                                        var randModel = myModels[Math.floor(Math.random() * myModels.length)];
                                        botui.message.add({
                                            content: "Hi I'm Holly. Please type 'evaluate' to stop chatting. What would you like to talk about?"
                                        })
                                        conversation(firstname, lastname, genre, randModel)
                                    }
                                    else {
                                        botui.message.add({
                                            content: 'Okay, bye for now!'
                                        })
                                    }
                                })
                            })
                        })
                    })
                })
            })
        })
    }


    function introduction() {
        // Holly introduces herself
        botui.message.add({
            content: "Hi, my name is Holly! I am a chatbot trained on movie scripts.",
        }).then(function () {
            botui.message.add({
                content: "Chat with me and I'll respond as if I'm an actor in your favorite movie. &#128540;"
            })
        }).then(function () {
            startChat()
        })
    }

    function startChat() {
        // Start chat
        botui.message.add({
            content: "What would you like to do?"
        }).then(function () {
            return botui.action.button({
                delay: 600,
                action: [
                    {
                        text: 'User testing',
                        value: 'usertest'
                    },
                    {
                        text: 'Play around',
                        value: 'playground'
                    }
                ]
            })
        }).then(function (res) {
            if (res.value == 'usertest') {
                userevaluation()
            }
        })
    }

    function continueChat() {
        return botui.action.button({
            delay: 600,
            action: [
                {
                    text: 'Chat again?',
                    value: 'continue'
                }
            ]
        }).then(function (res) {
            startChat();
        })
    }

    introduction();
});