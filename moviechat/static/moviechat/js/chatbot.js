$(document).ready(function () {
    // Initialization
    var botui = new BotUI('hello-world');

    // Start chat
    botui.message.add({
        content: 'Hi I am a chatbot!'
    })

    // Declare conversation function
    function conversation() {
        var stop = false;
        botui.action.text({
            action: {
                placeholder: ''
            }
        }).then(function (res) {
            if (res.value == 'goodbye') {
                stop = true;
                return false;
            }
            else {
                // send ajax request to nlp model
                $.ajax({
                    url: '/ajax/api_chat_response/',
                    data: {
                        'userinput': res.value
                    },
                    dataType: 'json',
                    success: function (data) {
                        botui.message.add({
                            content: data
                        });
                    },
                })
            }
        }).then(function () {
            // user would like to keep talking
            if (stop == false) {
                conversation()
            }
            // user would like to stop talking
            else {
                botui.message.add({
                    content: 'Goodbye, my friend!'
                });
            }
        })
    }

    conversation();
});